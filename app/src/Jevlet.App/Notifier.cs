using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Threading;

namespace Jevlet.App;

/// <summary>
/// Toasts in the corner of the screen, and the alarm, which rings until snoozed or stopped.
/// Drawn by the app (no notification platform dependency) in the palette's visual language.
/// </summary>
internal sealed class Notifier
{
    public enum Kind { Info, Reminder, Done, Error }

    private readonly List<Window> _toasts = [];
    private Window? _alarm;
    private MediaPlayer? _sound;

    public void Show(string title, string message, Kind kind = Kind.Info, TimeSpan? duration = null)
    {
        var toast = Build(title, message, kind, buttons: null);
        Present(toast);
        if (kind == Kind.Reminder)
        {
            System.Media.SystemSounds.Asterisk.Play();
        }
        var close = new DispatcherTimer { Interval = duration ?? TimeSpan.FromSeconds(kind == Kind.Reminder ? 30 : 5) };
        close.Tick += (_, _) =>
        {
            close.Stop();
            Dismiss(toast);
        };
        close.Start();
    }

    public void Ring(string label, Func<string> snooze, Func<string> stop)
    {
        StopRinging();
        _alarm = Build("Alarm", label, Kind.Reminder, [("Snooze 9 min", () => snooze()), ("Stop", () => stop())]);
        _alarm.Topmost = true;
        Present(_alarm);
        var sound = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "Media", "Alarm01.wav");
        if (File.Exists(sound))
        {
            _sound = new MediaPlayer();
            _sound.Open(new Uri(sound));
            _sound.MediaEnded += (_, _) =>
            {
                _sound?.Stop();
                _sound?.Play();
            };
            _sound.Play();
        }
        else
        {
            System.Media.SystemSounds.Exclamation.Play();
        }
    }

    public void StopRinging()
    {
        _sound?.Stop();
        _sound?.Close();
        _sound = null;
        if (_alarm is not null)
        {
            Dismiss(_alarm);
            _alarm = null;
        }
    }

    private Window Build(string title, string message, Kind kind, (string Label, Action Click)[]? buttons)
    {
        var accent = kind switch
        {
            Kind.Error => Theme.Brush("Danger"),
            Kind.Done => Theme.Brush("Success"),
            _ => Theme.Brush("Accent"),
        };
        var glyph = kind switch { Kind.Reminder => "", Kind.Done => "", Kind.Error => "", _ => "" };
        var stack = new StackPanel();
        var header = new StackPanel { Orientation = Orientation.Horizontal };
        header.Children.Add(new TextBlock { Text = glyph, FontFamily = Theme.Icons, FontSize = 16, Foreground = accent, Margin = new Thickness(0, 0, 10, 0), VerticalAlignment = VerticalAlignment.Center });
        header.Children.Add(new TextBlock { Text = title, FontSize = 13, FontWeight = FontWeights.SemiBold, Foreground = Theme.Brush("TextSecondary"), VerticalAlignment = VerticalAlignment.Center });
        stack.Children.Add(header);
        stack.Children.Add(new TextBlock { Text = message, FontSize = 16, TextWrapping = TextWrapping.Wrap, Foreground = Theme.Brush("TextPrimary"), Margin = new Thickness(0, 8, 0, 0) });
        var window = new Window
        {
            WindowStyle = WindowStyle.None,
            ResizeMode = ResizeMode.NoResize,
            ShowInTaskbar = false,
            Topmost = true,
            ShowActivated = false,
            SizeToContent = SizeToContent.Height,
            Width = 360,
            Background = Brushes.Transparent,
            FontFamily = Theme.Text,
        };
        if (buttons is not null)
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, 14, 0, 0) };
            foreach (var (label, click) in buttons)
            {
                var button = new Button { Content = label, Margin = new Thickness(8, 0, 0, 0), Padding = new Thickness(14, 6, 14, 6), MinWidth = 96 };
                button.Click += (_, _) => click();
                row.Children.Add(button);
            }
            stack.Children.Add(row);
        }
        else
        {
            window.MouseLeftButtonUp += (_, _) => Dismiss(window);
        }
        window.Content = new Border { Padding = new Thickness(18, 16, 18, 16), Child = stack };
        Backdrop.Apply(window, rounded: true);
        return window;
    }

    private void Present(Window window)
    {
        _toasts.Add(window);
        window.Opacity = 0;
        window.Show();
        Arrange();
        window.BeginAnimation(UIElement.OpacityProperty, new DoubleAnimation(1, TimeSpan.FromMilliseconds(160)));
    }

    private void Dismiss(Window window)
    {
        if (!_toasts.Remove(window))
        {
            return;
        }
        var fade = new DoubleAnimation(0, TimeSpan.FromMilliseconds(140));
        fade.Completed += (_, _) => window.Close();
        window.BeginAnimation(UIElement.OpacityProperty, fade);
        Arrange();
    }

    /// <summary>Stack toasts upward from the bottom-right corner of the work area.</summary>
    private void Arrange()
    {
        var area = SystemParameters.WorkArea;
        var bottom = area.Bottom - 16;
        foreach (var toast in Enumerable.Reverse(_toasts))
        {
            toast.UpdateLayout();
            toast.Left = area.Right - toast.Width - 16;
            toast.Top = bottom - toast.ActualHeight;
            bottom = toast.Top - 10;
        }
    }
}
