using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Jevlet.Core;

/// <summary>
/// Port of jevlet/assistant/text.py. Every list and pattern comes from shared/text_rules.json;
/// golden tests replay the Python outputs, so both sides offer the model identical options.
/// </summary>
public static class TextRules
{
    private const RegexOptions Ignore = RegexOptions.IgnoreCase | RegexOptions.CultureInvariant;

    private static readonly JsonElement Rules = LoadRules();
    public static readonly IReadOnlyList<string> Triggers = Strings("triggers");
    public static readonly IReadOnlyList<string> Stops = Strings("stops");
    public static readonly int TextCandidates = Rules.GetProperty("text_candidates").GetInt32();

    private static readonly Regex[] TriggerPatterns =
        Triggers.Select(t => new Regex($@"\b{Regex.Escape(t)}\b[:,]?\s+", RegexOptions.CultureInvariant)).ToArray();
    private static readonly Regex Word = new(Rules.GetProperty("word").GetString()!, RegexOptions.CultureInvariant);
    private static readonly Regex Courtesy = new(Rules.GetProperty("courtesy").GetString()!, Ignore);
    private static readonly Regex Delegate = new(Rules.GetProperty("delegate").GetString()!, Ignore);
    private static readonly Regex LabelBeforeNoun = new(Rules.GetProperty("label_before_noun").GetString()!, Ignore);
    private static readonly Regex Preposition = new(Rules.GetProperty("preposition").GetString()!, Ignore);
    private static readonly Regex PhraseEnd = new(Rules.GetProperty("phrase_end").GetString()!, Ignore);
    private static readonly Regex ProperRun = new(Rules.GetProperty("proper_run_case_sensitive").GetString()!, RegexOptions.CultureInvariant);
    public static readonly Regex TimeExpression = new(Rules.GetProperty("time_expression").GetString()!, Ignore);
    private static readonly Regex Quoted = new("[\"“']([^\"”']{1,200})[\"”']", RegexOptions.CultureInvariant);
    private static readonly Regex Labelled = new(@":\s+|\s+-\s+", RegexOptions.CultureInvariant);
    private static readonly Regex Possessive = new("'s$", RegexOptions.CultureInvariant);
    private static readonly Regex Duration = new(
        @"(\d+(?:\.\d+)?|[a-z]+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", RegexOptions.CultureInvariant);

    private static readonly (string Trigger, Regex Pattern, string[] Targets)[] Aliases = Rules.GetProperty("aliases")
        .EnumerateObject()
        .Select(p => (p.Name, new Regex($@"\b{Regex.Escape(p.Name)}\b", RegexOptions.CultureInvariant),
            p.Value.EnumerateArray().Select(v => v.GetString()!).ToArray()))
        .ToArray();
    private static readonly Dictionary<string, int> Units = Rules.GetProperty("duration_units")
        .EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32());
    private static readonly Dictionary<string, double> NumberWords = Rules.GetProperty("number_words")
        .EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetDouble());

    private static readonly char[] CleanChars = [' ', ',', ':', ';', '"', '\'', '“', '”'];
    private static readonly HashSet<string> NotProperRuns = ["I", "I'm", "I'll", "I'd"];

    private static JsonElement LoadRules()
    {
        using var document = Catalogue.ReadResource("text_rules.json");
        return document.RootElement.Clone();
    }

    private static List<string> Strings(string name) =>
        Rules.GetProperty(name).EnumerateArray().Select(v => v.GetString()!).ToList();

    /// <summary>Length-preserving lowercase (Python's <c>fold</c>): offsets stay valid.</summary>
    public static string Fold(string text)
    {
        var builder = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            // Python lowercases U+0130 to two code points; fold keeps it, and so do we.
            builder.Append(rune.Value == 0x130 ? rune : Rune.ToLowerInvariant(rune));
        }
        return builder.ToString();
    }

    private static string[] SplitWhitespace(string text) =>
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    public static IReadOnlyList<string> Words(string text) =>
        Word.Matches(Fold(text)).Select(m => m.Value).ToList();

    private static HashSet<string> Trigrams(string text)
    {
        // Code points, not UTF-16 units, as Python slices strings.
        var runes = $"  {Fold(text)} ".EnumerateRunes().Select(r => r.ToString()).ToArray();
        var grams = new HashSet<string>(StringComparer.Ordinal);
        for (var index = 0; index < runes.Length - 2; index++)
        {
            grams.Add(runes[index] + runes[index + 1] + runes[index + 2]);
        }
        return grams;
    }

    /// <summary>Blend of word overlap, character trigrams, and alias hits; 0 means unrelated.</summary>
    public static double Similarity(string query, string name)
    {
        var queryWords = Words(query).ToHashSet(StringComparer.Ordinal);
        var nameWords = Words(name).ToHashSet(StringComparer.Ordinal);
        if (nameWords.Count == 0)
        {
            return 0.0;
        }
        var overlap = (double)queryWords.Count(nameWords.Contains) / nameWords.Count;
        var gramsQuery = Trigrams(query);
        var gramsName = Trigrams(name);
        var trigram = (double)gramsName.Count(gramsQuery.Contains) / Math.Max(gramsName.Count, 1);
        var alias = 0.0;
        var loweredName = Fold(name);
        var loweredQuery = Fold(query);
        foreach (var (_, pattern, targets) in Aliases)
        {
            if (pattern.IsMatch(loweredQuery) && targets.Any(t => loweredName.Contains(t, StringComparison.Ordinal)))
            {
                alias = 0.8;
                break;
            }
        }
        return Math.Max(Math.Max(overlap, 0.6 * trigram), alias);
    }

    /// <summary>Top candidates by similarity; ties keep the caller's order.</summary>
    public static IReadOnlyList<string> Shortlist(string query, IEnumerable<string> names, int limit = 8) =>
        names.Select((name, index) => (Score: Similarity(query, name), Index: index, Name: name))
            .OrderByDescending(item => item.Score)
            .ThenBy(item => item.Index)
            .Take(limit)
            .Select(item => item.Name)
            .ToList();

    public static string StripCourtesy(string command) => Courtesy.Replace(command.Trim(), "", 1).Trim();

    private static List<string> WithCuts(string span)
    {
        var lowered = Fold(span);
        var cuts = new List<string>();
        foreach (var stop in Stops)
        {
            var at = lowered.IndexOf(stop, StringComparison.Ordinal);
            if (at > 0)
            {
                cuts.Add(span[..at].Trim());
            }
        }
        cuts.Add(span);
        return cuts;
    }

    private static List<string> PlacePhrases(string text)
    {
        var phrases = new List<string>();
        foreach (Match match in Preposition.Matches(text))
        {
            var rest = text[(match.Index + match.Length)..];
            var end = PhraseEnd.Match(rest);
            var phrase = (end.Success ? rest[..end.Index] : rest).Trim();
            var count = SplitWhitespace(phrase).Length;
            if (count is > 0 and <= 5)
            {
                phrases.Add(phrase);
            }
        }
        foreach (Match match in ProperRun.Matches(text))
        {
            var run = Possessive.Replace(match.Value.TrimEnd('.', ',', '\''), "");
            if (match.Index > 0 && !NotProperRuns.Contains(run))
            {
                phrases.Add(run);
            }
        }
        return phrases;
    }

    /// <summary>Plausible free-text arguments, all copied verbatim from the command.</summary>
    public static IReadOnlyList<string> SpanCandidates(string command, int? limit = null)
    {
        var text = StripCourtesy(command).TrimEnd(' ', '.', '?', '!');
        var candidates = new List<string>();
        candidates.AddRange(Quoted.Matches(text).Select(m => m.Groups[1].Value));
        candidates.AddRange(Delegate.Matches(text).Select(m => text[(m.Index + m.Length)..]).Take(2));
        var labelled = Labelled.Split(text, 2);
        if (labelled.Length == 2)
        {
            candidates.AddRange(WithCuts(labelled[1].Trim()));
        }
        candidates.AddRange(LabelBeforeNoun.Matches(text).Select(m => m.Groups[1].Value).Take(2));
        var untimed = string.Join(' ', SplitWhitespace(TimeExpression.Replace(text, " ")));
        string[] sources = untimed == text ? [text] : [text, untimed];
        foreach (var pattern in TriggerPatterns)
        {
            foreach (var source in sources)
            {
                foreach (Match match in pattern.Matches(Fold(source)))
                {
                    candidates.AddRange(WithCuts(source[(match.Index + match.Length)..].Trim()));
                }
            }
        }
        candidates.AddRange(PlacePhrases(text));
        var tokens = SplitWhitespace(text);
        candidates.AddRange(WithCuts(text));
        if (tokens.Length > 1)
        {
            candidates.AddRange(WithCuts(string.Join(' ', tokens[1..])));
        }
        foreach (var size in new[] { 1, 2, 3, 4 })
        {
            if (tokens.Length > size)
            {
                candidates.Add(string.Join(' ', tokens[^size..]));
            }
        }
        var unique = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var candidate in candidates)
        {
            var cleaned = candidate.Trim(CleanChars);
            if (cleaned.Length > 0 && seen.Add(Fold(cleaned)))
            {
                unique.Add(cleaned);
            }
        }
        return unique.Take(limit ?? TextCandidates).ToList();
    }

    public static bool HasTime(string command) => TimeExpression.IsMatch(command);

    /// <summary>Seconds for "25 minutes", "an hour and a half", "90s"; null if absent.</summary>
    public static int? ParseDuration(string command)
    {
        var text = Fold(command).Replace('-', ' ');
        var total = 0.0;
        var found = false;
        foreach (Match match in Duration.Matches(text))
        {
            var number = match.Groups[1].Value;
            double? value = char.IsDigit(number[0])
                ? double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : null
                : NumberWords.TryGetValue(number, out var word) ? word : null;
            if (value is null)
            {
                continue;
            }
            total += value.Value * Units[match.Groups[2].Value];
            found = true;
        }
        if (text.Contains("and a half", StringComparison.Ordinal) && found)
        {
            total += 0.5 * (text.Contains("hour", StringComparison.Ordinal) ? 3600 : 60);
        }
        return found && total > 0 ? (int)total : null;
    }
}
