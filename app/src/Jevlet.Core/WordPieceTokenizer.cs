using System.Globalization;
using System.Text;

namespace Jevlet.Core;

/// <summary>
/// BERT WordPiece tokenization, matching Hugging Face <c>tokenizers</c> for an uncased BERT
/// vocabulary (BertNormalizer, BertPreTokenizer, WordPiece). Golden tests compare every id.
/// </summary>
public sealed class WordPieceTokenizer
{
    private const int MaxInputCharsPerWord = 100;
    private readonly Dictionary<string, int> _vocab;
    private readonly bool _lowercase;

    public int UnknownId { get; }

    public WordPieceTokenizer(IReadOnlyList<string> vocabulary, bool lowercase = true)
    {
        _vocab = new Dictionary<string, int>(vocabulary.Count, StringComparer.Ordinal);
        for (var index = 0; index < vocabulary.Count; index++)
        {
            _vocab.TryAdd(vocabulary[index], index);
        }
        _lowercase = lowercase;
        UnknownId = _vocab["[UNK]"];
    }

    public static WordPieceTokenizer FromFile(string vocabPath, bool lowercase = true) =>
        new(File.ReadAllLines(vocabPath, Encoding.UTF8), lowercase);

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (var word in PreTokenize(Normalize(text)))
        {
            AppendWordPieces(word, ids);
        }
        return ids;
    }

    private string Normalize(string text)
    {
        var cleaned = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            if (rune.Value == 0 || rune.Value == 0xFFFD || IsControl(rune))
            {
                continue;
            }
            if (IsWhitespace(rune))
            {
                cleaned.Append(' ');
            }
            else if (IsChinese(rune.Value))
            {
                cleaned.Append(' ').Append(rune.ToString()).Append(' ');
            }
            else
            {
                cleaned.Append(rune.ToString());
            }
        }
        if (!_lowercase)
        {
            return cleaned.ToString();
        }
        // strip_accents defaults to lowercase: NFD, then drop non-spacing marks; then lowercase.
        var decomposed = cleaned.ToString().Normalize(NormalizationForm.FormD);
        var stripped = new List<Rune>(decomposed.Length);
        foreach (var rune in decomposed.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) != UnicodeCategory.NonSpacingMark)
            {
                stripped.Add(rune);
            }
        }
        return Lowercase(stripped);
    }

    private static string Lowercase(List<Rune> runes)
    {
        var builder = new StringBuilder(runes.Count);
        for (var index = 0; index < runes.Count; index++)
        {
            var rune = runes[index];
            if (rune.Value == 0x130)
            {
                builder.Append("i̇"); // full case mapping, as Rust's to_lowercase
            }
            else if (rune.Value == 0x3A3)
            {
                builder.Append(IsFinalSigma(runes, index) ? 'ς' : 'σ');
            }
            else
            {
                builder.Append(Rune.ToLowerInvariant(rune).ToString());
            }
        }
        return builder.ToString();
    }

    private static bool IsFinalSigma(List<Rune> runes, int index)
    {
        static bool Cased(Rune rune) => Rune.IsLetter(rune) && (Rune.IsUpper(rune) || Rune.IsLower(rune));
        var before = index > 0 && Cased(runes[index - 1]);
        var after = index + 1 < runes.Count && Cased(runes[index + 1]);
        return before && !after;
    }

    private static IEnumerable<string> PreTokenize(string text)
    {
        var word = new StringBuilder();
        foreach (var rune in text.EnumerateRunes())
        {
            if (Rune.IsWhiteSpace(rune))
            {
                if (word.Length > 0)
                {
                    yield return word.ToString();
                    word.Clear();
                }
            }
            else if (IsPunctuation(rune))
            {
                if (word.Length > 0)
                {
                    yield return word.ToString();
                    word.Clear();
                }
                yield return rune.ToString();
            }
            else
            {
                word.Append(rune.ToString());
            }
        }
        if (word.Length > 0)
        {
            yield return word.ToString();
        }
    }

    private void AppendWordPieces(string word, List<int> ids)
    {
        var runes = word.EnumerateRunes().Select(r => r.ToString()).ToArray();
        if (runes.Length > MaxInputCharsPerWord)
        {
            ids.Add(UnknownId);
            return;
        }
        var pieces = new List<int>();
        var start = 0;
        while (start < runes.Length)
        {
            var end = runes.Length;
            var found = -1;
            while (start < end)
            {
                var piece = string.Concat(runes[start..end]);
                if (start > 0)
                {
                    piece = "##" + piece;
                }
                if (_vocab.TryGetValue(piece, out var id))
                {
                    found = id;
                    break;
                }
                end--;
            }
            if (found < 0)
            {
                ids.Add(UnknownId);
                return;
            }
            pieces.Add(found);
            start = end;
        }
        ids.AddRange(pieces);
    }

    private static bool IsWhitespace(Rune rune) =>
        rune.Value is '\t' or '\n' or '\r' || Rune.IsWhiteSpace(rune);

    private static bool IsControl(Rune rune)
    {
        if (rune.Value is '\t' or '\n' or '\r')
        {
            return false;
        }
        return Rune.GetUnicodeCategory(rune) is UnicodeCategory.Control or UnicodeCategory.Format
            or UnicodeCategory.Surrogate or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned;
    }

    private static bool IsPunctuation(Rune rune)
    {
        if (rune.IsAscii && char.IsAsciiLetterOrDigit((char)rune.Value) is false
            && rune.Value is >= 33 and <= 126)
        {
            return true; // ASCII punctuation includes symbols such as $ + < = > ^ ` | ~
        }
        return Rune.GetUnicodeCategory(rune) is UnicodeCategory.ConnectorPunctuation
            or UnicodeCategory.DashPunctuation or UnicodeCategory.OpenPunctuation
            or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation
            or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;
    }

    private static bool IsChinese(int code) =>
        code is >= 0x4E00 and <= 0x9FFF or >= 0x3400 and <= 0x4DBF or >= 0x20000 and <= 0x2A6DF
            or >= 0x2A700 and <= 0x2B73F or >= 0x2B740 and <= 0x2B81F or >= 0x2B820 and <= 0x2CEAF
            or >= 0xF900 and <= 0xFAFF or >= 0x2F800 and <= 0x2FA1F;
}
