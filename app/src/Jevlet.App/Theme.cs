using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Shell;
using Microsoft.Win32;

namespace Jevlet.App;

/// <summary>Colours, type, and icons: follows the Windows light/dark setting and accent colour.</summary>
internal static class Theme
{
    public static readonly FontFamily Text = new("Segoe UI Variable Text, Segoe UI");
    public static readonly FontFamily Display = new("Segoe UI Variable Display, Segoe UI");
    public static readonly FontFamily Icons = new("Segoe Fluent Icons, Segoe MDL2 Assets");

    public static bool Dark { get; private set; } = true;

    public static Brush Brush(string key) => (Brush)Application.Current.Resources[key];

    /// <summary>Apply "System", "Dark", or "Light" to every brush the UI uses.</summary>
    public static void Apply(string mode)
    {
        Dark = mode switch
        {
            "Dark" => true,
            "Light" => false,
            _ => Registry.GetValue(@"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize", "AppsUseLightTheme", 1) is int light && light == 0,
        };
        var accent = AccentColor();
        var palette = Dark
            ? new Dictionary<string, Color>
            {
                ["Surface"] = Color.FromArgb(0xE6, 0x1C, 0x1C, 0x22),
                ["SurfaceSolid"] = Color.FromRgb(0x20, 0x20, 0x26),
                ["Card"] = Color.FromArgb(0x14, 0xFF, 0xFF, 0xFF),
                ["CardHover"] = Color.FromArgb(0x22, 0xFF, 0xFF, 0xFF),
                ["CardSelected"] = Color.FromArgb(0x2E, 0xFF, 0xFF, 0xFF),
                ["Stroke"] = Color.FromArgb(0x24, 0xFF, 0xFF, 0xFF),
                ["TextPrimary"] = Color.FromRgb(0xF4, 0xF4, 0xF6),
                ["TextSecondary"] = Color.FromRgb(0xB4, 0xB4, 0xBC),
                ["TextTertiary"] = Color.FromRgb(0x80, 0x80, 0x8A),
                ["Danger"] = Color.FromRgb(0xFF, 0x7A, 0x7A),
                ["Warning"] = Color.FromRgb(0xF5, 0xC2, 0x5B),
                ["Success"] = Color.FromRgb(0x5E, 0xE0, 0x96),
            }
            : new Dictionary<string, Color>
            {
                ["Surface"] = Color.FromArgb(0xE6, 0xF7, 0xF7, 0xFA),
                ["SurfaceSolid"] = Color.FromRgb(0xF6, 0xF6, 0xF9),
                ["Card"] = Color.FromArgb(0x0C, 0x00, 0x00, 0x00),
                ["CardHover"] = Color.FromArgb(0x14, 0x00, 0x00, 0x00),
                ["CardSelected"] = Color.FromArgb(0x1E, 0x00, 0x00, 0x00),
                ["Stroke"] = Color.FromArgb(0x1C, 0x00, 0x00, 0x00),
                ["TextPrimary"] = Color.FromRgb(0x17, 0x17, 0x1C),
                ["TextSecondary"] = Color.FromRgb(0x55, 0x55, 0x60),
                ["TextTertiary"] = Color.FromRgb(0x86, 0x86, 0x90),
                ["Danger"] = Color.FromRgb(0xC4, 0x2B, 0x1C),
                ["Warning"] = Color.FromRgb(0x9D, 0x5D, 0x00),
                ["Success"] = Color.FromRgb(0x0F, 0x7B, 0x0F),
            };
        palette["Accent"] = accent;
        palette["AccentSoft"] = Color.FromArgb(0x33, accent.R, accent.G, accent.B);
        foreach (var (key, color) in palette)
        {
            var brush = new SolidColorBrush(color);
            brush.Freeze();
            Application.Current.Resources[key] = brush;
        }
        Application.Current.ThemeMode = Dark ? ThemeMode.Dark : ThemeMode.Light;
    }

    private static Color AccentColor()
    {
        try
        {
            var settings = new Windows.UI.ViewManagement.UISettings();
            var value = settings.GetColorValue(Dark ? Windows.UI.ViewManagement.UIColorType.AccentLight2 : Windows.UI.ViewManagement.UIColorType.AccentDark1);
            return Color.FromRgb(value.R, value.G, value.B);
        }
        catch (Exception)
        {
            return Color.FromRgb(0x8B, 0x7C, 0xF8); // the icon's violet
        }
    }
}

/// <summary>Windows 11 acrylic behind a window, rounded corners, and a dark title frame.</summary>
internal static class Backdrop
{
    private static readonly bool HasSystemBackdrop = Environment.OSVersion.Version.Build >= 22621;

    public static void Apply(Window window, bool rounded)
    {
        window.Background = HasSystemBackdrop ? Brushes.Transparent : Theme.Brush("SurfaceSolid");
        if (HasSystemBackdrop)
        {
            // The DWM backdrop shows through a client area extended over the whole window.
            WindowChrome.SetWindowChrome(window, new WindowChrome
            {
                CaptionHeight = 0,
                GlassFrameThickness = new Thickness(-1),
                ResizeBorderThickness = new Thickness(0),
                UseAeroCaptionButtons = false,
            });
        }
        window.SourceInitialized += (_, _) => Refresh(window, rounded);
    }

    public static void Refresh(Window window, bool rounded)
    {
        var handle = new WindowInteropHelper(window).Handle;
        if (handle == 0)
        {
            return;
        }
        var dark = Theme.Dark ? 1 : 0;
        Native.DwmSetWindowAttribute(handle, Native.DWMWA_USE_IMMERSIVE_DARK_MODE, ref dark, sizeof(int));
        var corners = rounded ? 2 : 0; // DWMWCP_ROUND
        Native.DwmSetWindowAttribute(handle, Native.DWMWA_WINDOW_CORNER_PREFERENCE, ref corners, sizeof(int));
        if (HasSystemBackdrop)
        {
            var acrylic = 3; // DWMSBT_TRANSIENTWINDOW
            Native.DwmSetWindowAttribute(handle, Native.DWMWA_SYSTEMBACKDROP_TYPE, ref acrylic, sizeof(int));
            if (HwndSource.FromHwnd(handle) is { } source)
            {
                source.CompositionTarget.BackgroundColor = Colors.Transparent;
            }
        }
    }
}
