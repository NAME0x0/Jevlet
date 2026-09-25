namespace Jevlet.Core;

/// <summary>
/// What exists right now (Python's <c>Environment</c>): installed apps, open windows, and the
/// user's stores. Every argument option the model sees comes from here or from the command.
/// </summary>
public sealed record Snapshot(
    IReadOnlyList<string> Apps,
    IReadOnlyList<string> Windows,
    string CurrentWindow,
    IReadOnlyList<string> Events,
    IReadOnlyList<string> Alarms,
    IReadOnlyList<string> Todos,
    IReadOnlyList<string> Files,
    IReadOnlyList<string> Reminders)
{
    public static Snapshot Empty { get; } = new([], [], "", [], [], [], [], []);

    /// <summary>"process: title", with long titles shortened exactly as in training.</summary>
    public static string WindowLabel(string process, string title)
    {
        title = title.Length <= 60 ? title : title[..57] + "...";
        return string.IsNullOrEmpty(process) ? title : $"{process}: {title}";
    }
}

public static class SlotOptions
{
    /// <summary>Runtime options for one argument, always ending with "Not applicable".</summary>
    public static IReadOnlyList<string> For(string slot, string command, Snapshot snapshot)
    {
        var catalogue = Catalogue.Shared;
        IEnumerable<string> options = slot switch
        {
            "app" => TextRules.Shortlist(command, snapshot.Apps, 8),
            "window" => WindowOptions(command, snapshot),
            "event" => TextRules.Shortlist(command, snapshot.Events, 8),
            "alarm" => TextRules.Shortlist(command, snapshot.Alarms, 8),
            "todo" => TextRules.Shortlist(command, snapshot.Todos, 8),
            "file" => TextRules.Shortlist(command, snapshot.Files, 8),
            "reminder" => TextRules.Shortlist(command, snapshot.Reminders, 8),
            // Short fixed catalogues are offered whole: "pair my headphones" shares no words
            // with "Bluetooth and devices", so a lexical shortlist would drop the answer.
            "setting" or "website" or "folder" => catalogue.Targets[slot].Select(t => t.Key),
            "shortcut" => catalogue.Shortcuts.Select(s => s.Name),
            "text" => TextRules.SpanCandidates(command),
            _ when catalogue.FixedChoices.TryGetValue(slot, out var fixedChoices) => fixedChoices,
            _ => throw new KeyNotFoundException(slot),
        };
        var unique = new List<string>();
        foreach (var option in options)
        {
            if (option.Length > 0 && !unique.Contains(option, StringComparer.Ordinal))
            {
                unique.Add(option);
            }
        }
        unique.Add(catalogue.NotApplicable);
        return unique;
    }

    private static List<string> WindowOptions(string command, Snapshot snapshot)
    {
        var others = snapshot.Windows.Where(w => w != snapshot.CurrentWindow).ToList();
        var options = new List<string>();
        if (snapshot.CurrentWindow.Length > 0)
        {
            options.Add($"Current window ({snapshot.CurrentWindow})");
        }
        options.AddRange(TextRules.Shortlist(command, others, 7));
        return options;
    }
}
