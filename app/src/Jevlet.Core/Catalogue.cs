using System.Reflection;
using System.Text.Json;

namespace Jevlet.Core;

/// <summary>One action the assistant can take, as defined in shared/skills.json.</summary>
public sealed record Skill(
    string Key,
    string Name,
    string Description,
    IReadOnlyList<string> Slots,
    IReadOnlyList<string> OptionalSlots,
    double RiskFloor,
    string Title,
    bool NeedsScreen,
    string Time,
    string Icon);

/// <summary>The skill catalogue shared with the Python trainer (shared/skills.json).</summary>
public sealed class Catalogue
{
    public static Catalogue Shared { get; } = Load();

    public int Version { get; }
    public string SkillQuestion { get; }
    public string RiskQuestion { get; }
    public string NotApplicable { get; }
    public string StateFormat { get; }
    public IReadOnlyDictionary<string, string> SlotQuestions { get; }
    public IReadOnlyList<Skill> Skills { get; }
    public IReadOnlyDictionary<string, Skill> ByKey { get; }
    public IReadOnlyDictionary<string, Skill> ByName { get; }

    /// <summary>Settings pages, folders, and websites: display name to URI.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<KeyValuePair<string, string>>> Targets { get; }

    /// <summary>Keyboard shortcuts: display name to (modifiers, key).</summary>
    public IReadOnlyList<(string Name, IReadOnlyList<string> Modifiers, string Key)> Shortcuts { get; }

    /// <summary>Short closed lists offered whole (media, volume, power, ...).</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> FixedChoices { get; }

    private Catalogue(JsonElement root)
    {
        Version = root.GetProperty("version").GetInt32();
        SkillQuestion = root.GetProperty("skill_question").GetString()!;
        RiskQuestion = root.GetProperty("risk_question").GetString()!;
        NotApplicable = root.GetProperty("not_applicable").GetString()!;
        StateFormat = root.GetProperty("state_format").GetString()!;
        SlotQuestions = root.GetProperty("slots").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!);

        var catalogues = root.GetProperty("catalogues");
        var targets = new Dictionary<string, IReadOnlyList<KeyValuePair<string, string>>>();
        var fixedChoices = new Dictionary<string, IReadOnlyList<string>>();
        var shortcuts = new List<(string, IReadOnlyList<string>, string)>();
        foreach (var entry in catalogues.EnumerateObject())
        {
            if (entry.Name == "shortcut")
            {
                foreach (var shortcut in entry.Value.EnumerateObject())
                {
                    var modifiers = shortcut.Value[0].EnumerateArray().Select(m => m.GetString()!).ToList();
                    shortcuts.Add((shortcut.Name, modifiers, shortcut.Value[1].GetString()!));
                }
            }
            else if (entry.Value.ValueKind == JsonValueKind.Object)
            {
                targets[entry.Name] = entry.Value.EnumerateObject()
                    .Select(p => new KeyValuePair<string, string>(p.Name, p.Value.GetString()!)).ToList();
            }
            else
            {
                fixedChoices[entry.Name] = entry.Value.EnumerateArray().Select(v => v.GetString()!).ToList();
            }
        }
        Targets = targets;
        FixedChoices = fixedChoices;
        Shortcuts = shortcuts;

        Skills = root.GetProperty("skills").EnumerateArray().Select(ParseSkill).ToList();
        ByKey = Skills.ToDictionary(s => s.Key);
        ByName = Skills.ToDictionary(s => s.Name);
    }

    private static Skill ParseSkill(JsonElement row)
    {
        static IReadOnlyList<string> Strings(JsonElement row, string name) =>
            row.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Array
                ? value.EnumerateArray().Select(v => v.GetString()!).ToList()
                : [];
        static string Text(JsonElement row, string name) =>
            row.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString()! : "";

        return new Skill(
            Text(row, "key"),
            Text(row, "name"),
            Text(row, "description"),
            Strings(row, "slots"),
            Strings(row, "optional_slots"),
            row.TryGetProperty("risk_floor", out var floor) ? floor.GetDouble() : 0.0,
            Text(row, "title"),
            row.TryGetProperty("needs_screen", out var screen) && screen.GetBoolean(),
            Text(row, "time"),
            Text(row, "icon"));
    }

    public string State(string command, string window) =>
        StateFormat.Replace("{command}", command, StringComparison.Ordinal)
            .Replace("{window}", string.IsNullOrEmpty(window) ? "none" : window, StringComparison.Ordinal);

    internal static JsonDocument ReadResource(string name)
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream($"Jevlet.{name}")
            ?? throw new InvalidOperationException($"missing embedded resource {name}");
        return JsonDocument.Parse(stream);
    }

    private static Catalogue Load()
    {
        using var document = ReadResource("skills.json");
        return new Catalogue(document.RootElement);
    }
}
