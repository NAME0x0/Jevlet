using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Jevlet.Core;

/// <summary>What the model decided for one command, with the gate that governs running it.</summary>
public sealed class Plan(string command, Skill skill, double confidence, double risk)
{
    public const double SafeToRun = 0.9; // Jev's bar for destructive operations: P(safe) >= 0.9
    public const double Confident = 0.75; // skill and argument confidence needed for one Enter

    public string Command { get; } = command;
    public Skill Skill { get; } = skill;
    public double Confidence { get; } = confidence;
    public double Risk { get; set; } = Math.Max(risk, skill.RiskFloor);
    public Dictionary<string, string> Arguments { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, double> ArgumentConfidence { get; } = new(StringComparer.Ordinal);
    public List<(Skill Skill, double Probability)> Alternatives { get; } = [];
    public int? DurationSeconds { get; set; }
    public TimeMatch? When { get; set; }
    public string Missing { get; set; } = "";
    public double LatencyMs { get; set; }
    public DateTime Now { get; init; } = DateTime.Now;

    /// <summary>run: one Enter executes. confirm: a second Enter is required. clarify: cannot run.</summary>
    public string Gate
    {
        get
        {
            if (Skill.Key == "clarify" || Missing.Length > 0)
            {
                return "clarify";
            }
            var weakest = ArgumentConfidence.Values.Append(Confidence).Min();
            return 1.0 - Risk >= SafeToRun && weakest >= Confident ? "run" : "confirm";
        }
    }

    /// <summary>The skill's title with its arguments filled in, or the skill name.</summary>
    public string Title
    {
        get
        {
            var title = Skill.Title;
            foreach (var (slot, value) in Arguments)
            {
                title = title.Replace($"{{{slot}}}", Display(slot, value), StringComparison.Ordinal);
            }
            var when = DurationSeconds is { } seconds ? FormatDuration(seconds) : When?.Describe(Now) ?? "";
            if (when.Length > 0)
            {
                title = title.Replace("{when}", when, StringComparison.Ordinal);
            }
            title = title.Replace(" {when}", "", StringComparison.Ordinal);
            return title.Contains('{', StringComparison.Ordinal) ? Skill.Name : title;
        }
    }

    /// <summary>Human wording; the model's options keep the process name for precision.</summary>
    public static string Display(string slot, string value)
    {
        if (slot != "window")
        {
            return value;
        }
        if (value.StartsWith("Current window", StringComparison.Ordinal))
        {
            return "this window";
        }
        var split = value.IndexOf(": ", StringComparison.Ordinal);
        return split >= 0 ? value[(split + 2)..] : value;
    }

    public static string FormatDuration(int seconds)
    {
        var hours = seconds / 3600;
        var minutes = seconds % 3600 / 60;
        var rest = seconds % 60;
        var parts = new List<string>();
        if (hours > 0) parts.Add($"{hours} h");
        if (minutes > 0) parts.Add($"{minutes} min");
        if (rest > 0) parts.Add($"{rest} s");
        return parts.Count > 0 ? string.Join(' ', parts) : "?";
    }
}

/// <summary>
/// Command to plan (port of jevlet/assistant/planner.py). Phase 1 asks the skill and risk
/// questions together; phase 2 asks only the chosen skill's argument questions, with options
/// from the live snapshot. Times and durations are parsed, never predicted.
/// </summary>
public sealed class Planner(IDecisionModel engine)
{
    private static readonly Regex Steps = new(@"\s*(?:,\s*)?\b(?:and then|then|after that)\b\s*",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    public IDecisionModel Engine { get; } = engine;

    /// <summary>"open spotify then play the next song" becomes two steps.</summary>
    public static IReadOnlyList<string> SplitSteps(string command) =>
        Steps.Split(command.Trim()).Select(s => s.Trim(' ', ',', '.')).Where(s => s.Length > 0).ToList();

    public Plan Plan(string command, Snapshot snapshot, DateTime now)
    {
        var watch = Stopwatch.StartNew();
        var catalogue = Catalogue.Shared;
        var state = catalogue.State(command, snapshot.CurrentWindow);
        var answers = Engine.Evaluate(state,
        [
            new Question(catalogue.SkillQuestion, catalogue.Skills.Select(s => s.Name).ToList()),
            new Question(catalogue.RiskQuestion, ["True", "False"], "noul"),
        ]);
        var skills = answers[0];
        var ranked = skills.Options.Zip(skills.Probabilities).OrderByDescending(p => p.Second).ToList();
        var plan = PlanFor(catalogue.ByName[ranked[0].First], command, snapshot, ranked[0].Second, answers[1]["True"], now);
        plan.Alternatives.AddRange(ranked.Skip(1).Take(3).Select(p => (catalogue.ByName[p.First], p.Second)));
        plan.LatencyMs = watch.Elapsed.TotalMilliseconds;
        return plan;
    }

    /// <summary>Fill one skill's arguments (used for the top skill and for chosen alternatives).</summary>
    public Plan PlanFor(Skill skill, string command, Snapshot snapshot, double confidence, double risk, DateTime now)
    {
        var catalogue = Catalogue.Shared;
        var plan = new Plan(command, skill, confidence, risk) { Now = now };
        var slots = skill.Slots.Concat(skill.OptionalSlots).ToList();
        var questions = new List<(string Slot, Question Question)>();
        foreach (var slot in slots)
        {
            var options = SlotOptions.For(slot, command, snapshot);
            if (options.Count > 1)
            {
                questions.Add((slot, new Question(catalogue.SlotQuestions[slot], options)));
            }
            else if (skill.Slots.Contains(slot))
            {
                plan.Missing = slot;
            }
        }
        if (questions.Count > 0)
        {
            var answers = Engine.Evaluate(catalogue.State(command, snapshot.CurrentWindow), questions.Select(q => q.Question).ToList());
            for (var index = 0; index < questions.Count; index++)
            {
                var (slot, _) = questions[index];
                var answer = answers[index];
                if (answer.Selected == catalogue.NotApplicable)
                {
                    if (skill.Slots.Contains(slot))
                    {
                        plan.Missing = slot;
                    }
                    continue; // an optional argument the command did not give
                }
                plan.Arguments[slot] = answer.Selected;
                plan.ArgumentConfidence[slot] = answer.Confidence;
            }
        }
        switch (skill.Time)
        {
            case "duration":
                plan.DurationSeconds = TextRules.ParseDuration(command);
                if (plan.DurationSeconds is null)
                {
                    plan.Missing = "duration";
                }
                break;
            case "required":
                plan.When = TimeParser.Parse(command, now);
                if (plan.When is null)
                {
                    plan.Missing = "time";
                }
                break;
            case "optional":
                plan.When = TimeParser.Parse(command, now);
                break;
        }
        return plan;
    }
}
