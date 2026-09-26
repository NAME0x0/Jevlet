using System.Globalization;
using System.Net.Http;
using System.Text.Json;

namespace Jevlet.App;

/// <summary>Forecasts from Open-Meteo (free, no key, no account). Only the city name is sent.</summary>
internal static class Weather
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(6) };

    private static readonly Dictionary<int, string> Codes = new()
    {
        [0] = "Clear", [1] = "Mostly clear", [2] = "Partly cloudy", [3] = "Overcast", [45] = "Fog", [48] = "Freezing fog",
        [51] = "Light drizzle", [53] = "Drizzle", [55] = "Heavy drizzle", [61] = "Light rain", [63] = "Rain", [65] = "Heavy rain",
        [66] = "Freezing rain", [67] = "Freezing rain", [71] = "Light snow", [73] = "Snow", [75] = "Heavy snow", [77] = "Snow grains",
        [80] = "Showers", [81] = "Showers", [82] = "Violent showers", [85] = "Snow showers", [86] = "Snow showers",
        [95] = "Thunderstorm", [96] = "Thunderstorm with hail", [99] = "Thunderstorm with hail",
    };

    /// <summary>A headline and lines for today and the next days, or an error message.</summary>
    public static async Task<(bool Ok, string Headline, List<string> Lines)> Forecast(string place, DateTime? day)
    {
        try
        {
            var geo = await Http.GetStringAsync(
                $"https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name={Uri.EscapeDataString(place)}").ConfigureAwait(false);
            using var geoJson = JsonDocument.Parse(geo);
            if (!geoJson.RootElement.TryGetProperty("results", out var results) || results.GetArrayLength() == 0)
            {
                return (false, $"I couldn't find \"{place}\"", []);
            }
            var hit = results[0];
            var name = hit.GetProperty("name").GetString()!;
            var country = hit.TryGetProperty("country", out var c) ? c.GetString() : "";
            var lat = hit.GetProperty("latitude").GetDouble().ToString(CultureInfo.InvariantCulture);
            var lon = hit.GetProperty("longitude").GetDouble().ToString(CultureInfo.InvariantCulture);
            var body = await Http.GetStringAsync(
                $"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m" +
                "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=7").ConfigureAwait(false);
            using var json = JsonDocument.Parse(body);
            var current = json.RootElement.GetProperty("current");
            var daily = json.RootElement.GetProperty("daily");
            var dates = daily.GetProperty("time").EnumerateArray().Select(d => DateTime.Parse(d.GetString()!, CultureInfo.InvariantCulture)).ToList();
            var lines = new List<string>();
            for (var index = 0; index < dates.Count; index++)
            {
                if (day is { } wanted && dates[index].Date < wanted.Date)
                {
                    continue;
                }
                var label = index == 0 ? "Today" : index == 1 ? "Tomorrow" : dates[index].ToString("dddd", CultureInfo.InvariantCulture);
                lines.Add($"{label}: {Describe(daily.GetProperty("weather_code")[index].GetInt32())}, " +
                    $"{daily.GetProperty("temperature_2m_min")[index].GetDouble():0}–{daily.GetProperty("temperature_2m_max")[index].GetDouble():0} °C, " +
                    $"rain {daily.GetProperty("precipitation_probability_max")[index].GetInt32()}%");
                if (lines.Count == 3)
                {
                    break;
                }
            }
            var headline = $"{name}{(string.IsNullOrEmpty(country) ? "" : $", {country}")}: {current.GetProperty("temperature_2m").GetDouble():0} °C, " +
                $"{Describe(current.GetProperty("weather_code").GetInt32()).ToLowerInvariant()} (feels {current.GetProperty("apparent_temperature").GetDouble():0} °C)";
            return (true, headline, lines);
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException or JsonException or KeyNotFoundException)
        {
            Log.Error("weather failed", error);
            return (false, "Weather is unavailable right now (no connection?)", []);
        }
    }

    private static string Describe(int code) => Codes.TryGetValue(code, out var text) ? text : "Unsettled";

    /// <summary>A sensible default city from the Windows time zone, until the user sets one.</summary>
    public static string DefaultCity()
    {
        var zone = TimeZoneInfo.Local;
        if (TimeZoneInfo.TryConvertWindowsIdToIanaId(zone.Id, out var iana) && iana.Contains('/', StringComparison.Ordinal))
        {
            return iana[(iana.LastIndexOf('/') + 1)..].Replace('_', ' ');
        }
        return "London";
    }
}
