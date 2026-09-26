using System.Globalization;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Threading;
using Jevlet.Core;

namespace Jevlet.App;

/// <summary>
/// The command palette: type, see what Jevlet will do and how sure it is, press Enter.
/// Planning runs off the UI thread on every pause in typing; results for stale text are dropped.
/// </summary>
public partial class PaletteWindow : Window
{
    // Skills whose result is an answer to read, so the palette stays open to show it.
    private static readonly HashSet<string> Answers =
    [
        "list_alarms", "list_reminders", "show_agenda", "show_todos", "show_notes", "calculate", "world_time",
        "weather", "today", "battery",
    ];
    // Skills that change the user's stores or timers: confirm with a toast once done.
    private static readonly HashSet<string> Confirmations =
    [
        "set_alarm", "cancel_alarm", "set_reminder", "cancel_reminder", "create_event", "move_event", "cancel_event",
        "add_todo", "complete_todo", "take_note", "timer", "timer_control", "stopwatch", "screenshot", "alarm_control",
    ];
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    private readonly Services _services;
    private readonly DispatcherTimer _debounce = new() { Interval = TimeSpan.FromMilliseconds(70) };
    // Keeps a laptop GPU out of its low-power state while the palette is open (see KeepWarm).
    private readonly DispatcherTimer _warm = new() { Interval = TimeSpan.FromMilliseconds(300) };
    private int _warming;
    private int _version;
    private CommandContext _context = new(0, [], new Dictionary<string, long>());
    private Snapshot _snapshot = Snapshot.Empty;
    private List<Plan?> _choices = [];
    private int _selected;
    private bool _confirming;
    private bool _running;

    internal PaletteWindow(Services services)
    {
        _services = services;
        InitializeComponent();
        Backdrop.Apply(this, rounded: true);
        _debounce.Tick += async (_, _) => await PlanNow().ConfigureAwait(true);
        _warm.Tick += (_, _) => KeepWarm();
        Input.TextChanged += (_, _) => OnTextChanged();
        PreviewKeyDown += OnKey;
        Deactivated += (_, _) =>
        {
            if (!_running)
            {
                Dismiss();
            }
        };
    }

    // ------------------------------------------------------------------ show and hide

    public void Toggle()
    {
        if (IsVisible && IsActive)
        {
            Dismiss();
        }
        else
        {
            Summon();
        }
    }

    public void Summon(string? text = null)
    {
        var previous = Native.GetForegroundWindow();
        var own = new WindowInteropHelper(this).Handle;
        if (previous == own)
        {
            previous = _context.Previous;
        }
        Capture(previous, own);
        Position(previous);
        _confirming = false;
        KeepWarm();
        _warm.Start();
        Show();
        Activate();
        Native.SetForegroundWindow(new WindowInteropHelper(this).Handle);
        Input.Focus();
        if (text is not null)
        {
            Input.Text = text;
            Input.CaretIndex = text.Length;
        }
        else
        {
            Input.SelectAll();
        }
        if (Input.Text.Trim().Length == 0)
        {
            RenderToday();
        }
        UpdateModelInfo();
    }

    public void Dismiss()
    {
        _debounce.Stop();
        _warm.Stop();
        Hide();
    }

    private void KeepWarm()
    {
        if (_services.Engine is not { } engine || Interlocked.Exchange(ref _warming, 1) == 1)
        {
            return;
        }
        Task.Run(() =>
        {
            try
            {
                engine.KeepWarm();
            }
            catch (Exception error) when (error is not OutOfMemoryException)
            {
                Log.Error("keep-warm pass failed", error);
            }
            finally
            {
                Interlocked.Exchange(ref _warming, 0);
            }
        });
    }

    /// <summary>What is on screen and in the stores right now (the palette itself excluded).</summary>
    private void Capture(nint previous, nint own)
    {
        var windows = Desktop.Windows([own]);
        var current = windows.FirstOrDefault(w => w.Handle == previous) ?? (windows.Count > 0 ? windows[0] : null);
        var snapshot = Snapshot.Empty with
        {
            Apps = Desktop.Apps.Select(a => a.Name).ToList(),
            Windows = windows.Select(w => w.Label).ToList(),
            CurrentWindow = current?.Label ?? "",
        };
        var (filled, ids) = _services.Stores.Fill(snapshot, DateTime.Now);
        _snapshot = filled;
        _context = new CommandContext(current?.Handle ?? previous, windows, ids);
    }

    /// <summary>Upper third of the monitor the user is working on.</summary>
    private void Position(nint previous)
    {
        var screen = System.Windows.Forms.Screen.FromHandle(previous != 0 ? previous : new WindowInteropHelper(this).Handle);
        var dpi = VisualTreeHelper.GetDpi(this);
        var area = screen.WorkingArea;
        Left = area.Left / dpi.DpiScaleX + (area.Width / dpi.DpiScaleX - Width) / 2;
        Top = area.Top / dpi.DpiScaleY + area.Height / dpi.DpiScaleY * 0.2;
    }

    // ------------------------------------------------------------------ planning

    private void OnTextChanged()
    {
        Placeholder.Visibility = Input.Text.Length == 0 ? Visibility.Visible : Visibility.Collapsed;
        _version++;
        _confirming = false;
        _debounce.Stop();
        if (Input.Text.Trim().Length == 0)
        {
            _choices = [];
            RenderToday();
            return;
        }
        _debounce.Start();
    }

    private async Task PlanNow()
    {
        _debounce.Stop();
        var text = Input.Text.Trim();
        var version = _version;
        var planner = _services.Planner;
        if (planner is null)
        {
            RenderMessage("", _services.ModelProblem ?? "Loading the model…", "TextSecondary");
            return;
        }
        var steps = Planner.SplitSteps(text);
        var step = steps.Count > 0 ? steps[0] : text;
        var snapshot = _snapshot;
        Status.Text = "…";
        Plan plan;
        try
        {
            plan = await Task.Run(() => PlanWithFiles(planner, step, snapshot, DateTime.Now)).ConfigureAwait(true);
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Log.Error("planning failed", error);
            RenderMessage("", "Something went wrong while thinking. It's in the log.", "Danger");
            return;
        }
        if (version != _version)
        {
            return; // the text changed while we were planning
        }
        _choices = [plan, .. plan.Alternatives.Select(_ => (Plan?)null)];
        _selected = 0;
        Status.Text = $"{plan.LatencyMs:0} ms";
        RenderPlan();
    }

    private static Plan PlanWithFiles(Planner planner, string command, Snapshot snapshot, DateTime now)
    {
        var plan = planner.Plan(command, snapshot, now);
        if (plan.Skill.Slots.Contains("file"))
        {
            // Files are searched only when the model asks for one: the index answers in milliseconds.
            var files = FileSearch.Candidates(command);
            if (files.Count > 0)
            {
                var withFiles = planner.PlanFor(plan.Skill, command, snapshot with { Files = files }, plan.Confidence, plan.Risk, now);
                withFiles.Alternatives.AddRange(plan.Alternatives);
                withFiles.LatencyMs = plan.LatencyMs;
                return withFiles;
            }
        }
        return plan;
    }

    private async Task Select(int index)
    {
        if (_choices.Count == 0)
        {
            return;
        }
        _selected = Math.Clamp(index, 0, _choices.Count - 1);
        _confirming = false;
        if (_choices[_selected] is null && _choices[0] is { } top && _services.Planner is { } planner)
        {
            var (skill, probability) = top.Alternatives[_selected - 1];
            var snapshot = _snapshot;
            var version = _version;
            var chosen = _selected;
            var plan = await Task.Run(() => planner.PlanFor(skill, top.Command, snapshot, probability, top.Risk, DateTime.Now)).ConfigureAwait(true);
            if (version != _version || chosen >= _choices.Count)
            {
                return;
            }
            _choices[chosen] = plan;
        }
        RenderPlan();
    }

    // ------------------------------------------------------------------ keys and running

    private async void OnKey(object sender, KeyEventArgs e)
    {
        switch (e.Key)
        {
            case Key.Escape:
                e.Handled = true;
                Dismiss();
                break;
            case Key.Down:
                e.Handled = true;
                await Select(_selected + 1).ConfigureAwait(true);
                break;
            case Key.Up:
                e.Handled = true;
                await Select(_selected - 1).ConfigureAwait(true);
                break;
            case Key.Enter:
                e.Handled = true;
                await Run().ConfigureAwait(true);
                break;
        }
    }

    private async Task Run()
    {
        if (_running)
        {
            return;
        }
        if (_debounce.IsEnabled)
        {
            await PlanNow().ConfigureAwait(true); // Enter right after typing: plan first
        }
        if (_selected >= _choices.Count || _choices[_selected] is not { } plan)
        {
            return;
        }
        if (plan.Gate == "clarify")
        {
            Shake();
            return;
        }
        if (plan.Gate == "confirm" && !_confirming)
        {
            _confirming = true;
            RenderPlan();
            return;
        }
        _running = true;
        try
        {
            var steps = Planner.SplitSteps(Input.Text.Trim());
            await Execute(plan, _selected).ConfigureAwait(true);
            foreach (var step in steps.Skip(1))
            {
                // Later steps are re-planned against the screen the earlier steps produced.
                await Task.Delay(500).ConfigureAwait(true);
                Capture(Native.GetForegroundWindow(), new WindowInteropHelper(this).Handle);
                var planner = _services.Planner!;
                var snapshot = _snapshot;
                var next = await Task.Run(() => PlanWithFiles(planner, step, snapshot, DateTime.Now)).ConfigureAwait(true);
                if (next.Gate != "run")
                {
                    Summon(step); // stop and show what needs a decision
                    return;
                }
                await Execute(next, 0).ConfigureAwait(true);
            }
        }
        finally
        {
            _running = false;
        }
    }

    private async Task Execute(Plan plan, int rank)
    {
        var answer = Answers.Contains(plan.Skill.Key);
        if (!answer)
        {
            Dismiss(); // give focus back to the window the action is for
        }
        else
        {
            RenderMessage("", "Working…", "TextSecondary");
        }
        var outcome = await _services.Executor.Run(plan, _context).ConfigureAwait(true);
        _services.Stores.Record(plan.Command, plan.Skill.Key, JsonSerializer.Serialize(plan.Arguments), plan.Gate,
            outcome.Ok ? "ok" : $"failed: {outcome.Message}", rank);
        Log.Info($"ran {plan.Skill.Key} ({outcome.Ok}): {outcome.Message}");
        if (answer)
        {
            RenderOutcome(outcome);
        }
        else if (!outcome.Ok)
        {
            _services.Notifier.Show("Couldn't do that", outcome.Message, Notifier.Kind.Error);
        }
        else if (Confirmations.Contains(plan.Skill.Key))
        {
            _services.Notifier.Show(plan.Skill.Name, outcome.Message, Notifier.Kind.Done);
        }
    }

    private void Shake()
    {
        var animation = new System.Windows.Media.Animation.DoubleAnimationUsingKeyFrames { Duration = TimeSpan.FromMilliseconds(280) };
        foreach (var (offset, ms) in new[] { (-8.0, 40), (8.0, 100), (-5.0, 160), (5.0, 220), (0.0, 280) })
        {
            animation.KeyFrames.Add(new System.Windows.Media.Animation.LinearDoubleKeyFrame(offset, System.Windows.Media.Animation.KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(ms))));
        }
        var shift = new TranslateTransform();
        Frame.RenderTransform = shift;
        shift.BeginAnimation(TranslateTransform.XProperty, animation);
    }

    // ------------------------------------------------------------------ rendering

    private void UpdateModelInfo() =>
        ModelInfo.Text = _services.Planner is null ? "loading model" : _services.ModelLabel;

    private static string Glyph(Skill skill) =>
        int.TryParse(skill.Icon, NumberStyles.HexNumber, Invariant, out var code) ? char.ConvertFromUtf32(code) : "";

    private static Brush Brush(string key) => Theme.Brush(key);

    private static Border Row(string glyph, string title, string? subtitle, UIElement? trailing, bool selected, bool large = false,
        Brush? iconBrush = null, Brush? border = null)
    {
        var grid = new Grid();
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var icon = new Border
        {
            Width = large ? 40 : 32,
            Height = large ? 40 : 32,
            CornerRadius = new CornerRadius(8),
            Background = selected ? Brush("AccentSoft") : Brush("Card"),
            Child = new TextBlock
            {
                Text = glyph,
                FontFamily = Theme.Icons,
                FontSize = large ? 18 : 15,
                Foreground = iconBrush ?? (selected ? Brush("Accent") : Brush("TextSecondary")),
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center,
            },
            VerticalAlignment = VerticalAlignment.Center,
        };
        grid.Children.Add(icon);
        var text = new StackPanel { Margin = new Thickness(12, 0, 12, 0), VerticalAlignment = VerticalAlignment.Center };
        text.Children.Add(new TextBlock
        {
            Text = title,
            FontSize = large ? 16 : 14,
            FontWeight = large ? FontWeights.SemiBold : FontWeights.Normal,
            Foreground = Brush("TextPrimary"),
            TextTrimming = TextTrimming.CharacterEllipsis,
        });
        if (!string.IsNullOrEmpty(subtitle))
        {
            text.Children.Add(new TextBlock { Text = subtitle, FontSize = 12, Foreground = Brush("TextSecondary"), Margin = new Thickness(0, 2, 0, 0), TextTrimming = TextTrimming.CharacterEllipsis });
        }
        Grid.SetColumn(text, 1);
        grid.Children.Add(text);
        if (trailing is not null)
        {
            Grid.SetColumn(trailing, 2);
            grid.Children.Add(trailing);
        }
        return new Border
        {
            Child = grid,
            Padding = new Thickness(10, large ? 10 : 7, 12, large ? 10 : 7),
            Margin = new Thickness(0, 0, 0, 2),
            CornerRadius = new CornerRadius(8),
            Background = selected ? Brush("CardSelected") : Brushes.Transparent,
            BorderBrush = border ?? Brushes.Transparent,
            BorderThickness = new Thickness(border is null ? 0 : 1),
        };
    }

    private static Border Chip(string text, string brush, bool filled = false) => new()
    {
        CornerRadius = new CornerRadius(9),
        Padding = new Thickness(9, 3, 9, 3),
        Margin = new Thickness(6, 0, 0, 0),
        Background = filled ? Brush("AccentSoft") : Brush("Card"),
        VerticalAlignment = VerticalAlignment.Center,
        Child = new TextBlock { Text = text, FontSize = 11, FontWeight = FontWeights.SemiBold, Foreground = Brush(brush) },
    };

    private static string MissingHint(Plan plan) => plan.Skill.Key == "clarify"
        ? plan.Risk >= 0.5 ? "That could delete, spend, or share something. I won't do it from here." : "I'm not sure what you want. Try saying it another way."
        : plan.Missing switch
        {
            "time" => "When? Add a time, like “tomorrow at 9”.",
            "duration" => "For how long? Add a length, like “25 minutes”.",
            "control" => "Clicking on screen arrives in the next update.",
            var slot => $"Which {slot.Replace('_', ' ')}? Say it in the command.",
        };

    private void RenderPlan()
    {
        Body.Children.Clear();
        if (_choices.Count == 0 || _choices[0] is not { } top)
        {
            return;
        }
        var shown = _choices[_selected] ?? top;
        var steps = Planner.SplitSteps(Input.Text.Trim());
        var gate = shown.Gate;
        Border trailing = gate switch
        {
            "run" => Chip("Enter to run", "Accent", filled: true),
            "confirm" => Chip(_confirming ? "Enter again to run" : "Check, then Enter twice", "Warning"),
            _ => Chip("Needs detail", "TextSecondary"),
        };
        var sure = $"{shown.Confidence:P0} sure";
        var risk = shown.Risk >= 0.5 ? " · risky" : shown.Risk >= 0.1 ? " · careful" : "";
        var subtitle = gate == "clarify" ? MissingHint(shown) : $"{shown.Skill.Name} · {sure}{risk}";
        var card = Row(Glyph(shown.Skill), shown.Title, subtitle, trailing, selected: true, large: true,
            border: _confirming ? Brush("Warning") : null);
        Body.Children.Add(card);
        if (steps.Count > 1)
        {
            Body.Children.Add(new TextBlock
            {
                Text = "Then: " + string.Join("  →  ", steps.Skip(1)),
                FontSize = 12,
                Foreground = Brush("TextSecondary"),
                Margin = new Thickness(62, 0, 12, 8),
                TextTrimming = TextTrimming.CharacterEllipsis,
            });
        }
        if (_confirming)
        {
            Body.Children.Add(new TextBlock
            {
                Text = "This can change or remove something. Press Enter again to go ahead, or Esc to stop.",
                FontSize = 12,
                Foreground = Brush("Warning"),
                Margin = new Thickness(62, 0, 12, 8),
                TextWrapping = TextWrapping.Wrap,
            });
        }
        if (top.Alternatives.Count > 0)
        {
            Body.Children.Add(new TextBlock { Text = "OR", FontSize = 11, FontWeight = FontWeights.SemiBold, Foreground = Brush("TextTertiary"), Margin = new Thickness(12, 8, 0, 4) });
            for (var index = 0; index < top.Alternatives.Count; index++)
            {
                var (skill, probability) = top.Alternatives[index];
                var isShown = _selected == index + 1;
                var title = _choices[index + 1]?.Title ?? skill.Name;
                Body.Children.Add(Row(Glyph(skill), title, null, Chip($"{probability:P0}", "TextTertiary"), selected: false,
                    border: isShown ? Brush("Stroke") : null));
            }
            if (_selected > 0)
            {
                // The top suggestion moves into the list when an alternative is shown.
                Body.Children.Insert(Body.Children.Count - top.Alternatives.Count,
                    Row(Glyph(top.Skill), top.Title, null, Chip($"{top.Confidence:P0}", "TextTertiary"), selected: false));
            }
        }
        EnterHint.Text = gate == "clarify" ? "needs detail" : _confirming ? "confirm" : "run";
    }

    private void RenderOutcome(Outcome outcome)
    {
        Body.Children.Clear();
        Body.Children.Add(Row(outcome.Ok ? "" : "", outcome.Message, null, null, selected: true, large: true,
            iconBrush: Brush(outcome.Ok ? "Success" : "Danger")));
        foreach (var line in outcome.Lines ?? [])
        {
            Body.Children.Add(new TextBlock
            {
                Text = line,
                FontSize = 13,
                Foreground = Brush("TextSecondary"),
                Margin = new Thickness(64, 2, 12, 2),
                TextWrapping = TextWrapping.Wrap,
            });
        }
        Body.Children.Add(new Border { Height = 6 });
        EnterHint.Text = "run";
    }

    private void RenderMessage(string glyph, string message, string brush)
    {
        Body.Children.Clear();
        Body.Children.Add(Row(glyph, message, null, null, selected: false, iconBrush: Brush(brush)));
    }

    /// <summary>The empty palette: a glance at the day.</summary>
    private void RenderToday()
    {
        Body.Children.Clear();
        var now = DateTime.Now;
        var header = new Grid { Margin = new Thickness(12, 4, 12, 8) };
        header.Children.Add(new TextBlock { Text = now.ToString("dddd d MMMM", Invariant).ToUpperInvariant(), FontSize = 11, FontWeight = FontWeights.SemiBold, Foreground = Brush("TextTertiary") });
        header.Children.Add(new TextBlock { Text = now.ToString("h:mm tt", Invariant), FontSize = 11, FontWeight = FontWeights.SemiBold, Foreground = Brush("TextTertiary"), HorizontalAlignment = HorizontalAlignment.Right });
        Body.Children.Add(header);
        var rows = new List<Border>();
        var stores = _services.Stores;
        foreach (var item in stores.Events(now, now.Date.AddDays(2)).Take(3))
        {
            rows.Add(Row("", Capitalize(item.Title), new TimeMatch(item.Starts, true, true).Describe(now), null, false));
        }
        foreach (var reminder in stores.PendingReminders().Take(3))
        {
            rows.Add(Row("", Capitalize(reminder.Task), $"Reminder · {new TimeMatch(reminder.Due, true, true).Describe(now)}", null, false));
        }
        foreach (var timer in _services.Scheduler.Timers)
        {
            rows.Add(Row("", timer.Label, $"{Plan.FormatDuration(Math.Max(1, (int)timer.Remaining.TotalSeconds))} left{(timer.PausedRemaining is null ? "" : " · paused")}", null, false));
        }
        var alarms = stores.Alarms().Where(a => a.Enabled).ToList();
        if (alarms.Count > 0)
        {
            var next = alarms.OrderBy(a => (a.Time - now.TimeOfDay + TimeSpan.FromDays(1)).TotalMinutes % 1440).First();
            rows.Add(Row("", $"Alarm {DateTime.Today.Add(next.Time).ToString("h:mm tt", Invariant)}", next.Label.Length > 0 ? Capitalize(next.Label) : null, null, false));
        }
        var todos = stores.OpenTodos().Count;
        if (todos > 0)
        {
            rows.Add(Row("", $"{todos} open to-do{(todos == 1 ? "" : "s")}", "Say “show my to-dos”", null, false));
        }
        if (rows.Count == 0)
        {
            rows.Add(Row("", "Your day is clear", "Try “remind me to stretch in 30 minutes” or “open spotify”", null, false));
        }
        foreach (var row in rows.Take(7))
        {
            Body.Children.Add(row);
        }
        Status.Text = _services.Planner is null ? "loading" : "ready";
        EnterHint.Text = "run";
    }

    private static string Capitalize(string text) => text.Length == 0 ? text : char.ToUpperInvariant(text[0]) + text[1..];
}
