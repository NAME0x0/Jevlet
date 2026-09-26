using System.Globalization;
using System.Text.RegularExpressions;

namespace Jevlet.Core;

/// <summary>
/// Arithmetic and unit conversion for the calculate skill: "18 percent of 240", "12 times 37",
/// "the square root of 144", "26 miles in km", "100 fahrenheit to celsius". Code computes;
/// the model only picks the span.
/// </summary>
public static partial class Calculator
{
    private static readonly (string Word, string Symbol)[] Words =
    [
        ("multiplied by", "*"), ("times", "*"), ("divided by", "/"), ("over", "/"), ("plus", "+"), ("minus", "-"),
        ("to the power of", "^"), ("squared", "^2"), ("cubed", "^3"), ("x", "*"), ("×", "*"), ("÷", "/"),
    ];

    // Factor to a base unit per dimension; temperature is handled separately.
    private static readonly Dictionary<string, (string Dimension, double Factor)> Units = new(StringComparer.Ordinal)
    {
        ["mm"] = ("length", 0.001), ["cm"] = ("length", 0.01), ["m"] = ("length", 1), ["meter"] = ("length", 1), ["metre"] = ("length", 1),
        ["km"] = ("length", 1000), ["kilometer"] = ("length", 1000), ["kilometre"] = ("length", 1000),
        ["in"] = ("length", 0.0254), ["inch"] = ("length", 0.0254), ["ft"] = ("length", 0.3048), ["foot"] = ("length", 0.3048),
        ["feet"] = ("length", 0.3048), ["yd"] = ("length", 0.9144), ["yard"] = ("length", 0.9144), ["mile"] = ("length", 1609.344), ["mi"] = ("length", 1609.344),
        ["mg"] = ("mass", 1e-6), ["g"] = ("mass", 0.001), ["gram"] = ("mass", 0.001), ["kg"] = ("mass", 1), ["kilogram"] = ("mass", 1),
        ["lb"] = ("mass", 0.45359237), ["pound"] = ("mass", 0.45359237), ["oz"] = ("mass", 0.028349523125), ["ounce"] = ("mass", 0.028349523125),
        ["stone"] = ("mass", 6.35029318), ["ton"] = ("mass", 1000), ["tonne"] = ("mass", 1000),
        ["ml"] = ("volume", 0.001), ["l"] = ("volume", 1), ["liter"] = ("volume", 1), ["litre"] = ("volume", 1),
        ["gallon"] = ("volume", 3.785411784), ["cup"] = ("volume", 0.2365882365), ["pint"] = ("volume", 0.473176473), ["floz"] = ("volume", 0.0295735295625),
        ["mph"] = ("speed", 0.44704), ["km/h"] = ("speed", 1 / 3.6), ["kmh"] = ("speed", 1 / 3.6), ["kph"] = ("speed", 1 / 3.6), ["m/s"] = ("speed", 1), ["knot"] = ("speed", 0.514444),
        ["b"] = ("data", 1), ["byte"] = ("data", 1), ["kb"] = ("data", 1e3), ["mb"] = ("data", 1e6), ["gb"] = ("data", 1e9), ["tb"] = ("data", 1e12),
        ["second"] = ("time", 1), ["sec"] = ("time", 1), ["minute"] = ("time", 60), ["min"] = ("time", 60), ["hour"] = ("time", 3600),
        ["hr"] = ("time", 3600), ["day"] = ("time", 86400), ["week"] = ("time", 604800), ["year"] = ("time", 31557600),
        ["acre"] = ("area", 4046.8564224), ["hectare"] = ("area", 10000), ["square meter"] = ("area", 1), ["sqm"] = ("area", 1), ["square feet"] = ("area", 0.09290304),
    };

    private static readonly string[] Temperatures = ["celsius", "c", "fahrenheit", "f", "kelvin", "k"];

    [GeneratedRegex(@"^(?:convert\s+)?(-?\d+(?:[.,]\d+)?)\s*([a-z/ ]+?)\s+(?:in|to|into|as)\s+([a-z/ ]+?)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex ConversionPattern();

    [GeneratedRegex(@"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex PercentOfPattern();

    [GeneratedRegex(@"(?:the\s+)?square root of\s+", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex RootPattern();

    /// <summary>The result as text ("43.2", "41.84 km"), or null if the expression is not understood.</summary>
    public static string? Evaluate(string expression)
    {
        var text = expression.Trim().TrimEnd('?', '.', '=').Trim();
        text = Regex.Replace(text, @"^(what's|what is|calculate|compute|work out|how much is)\s+", "", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant).Trim();
        text = Regex.Replace(text, @"\s+equals$", "", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        var conversion = Convert(text);
        if (conversion is not null)
        {
            return conversion;
        }
        text = PercentOfPattern().Replace(text, m => $"({m.Groups[1].Value}/100*{m.Groups[2].Value})");
        text = RootPattern().Replace(text, "sqrt ");
        foreach (var (word, symbol) in Words)
        {
            text = Regex.Replace(text, $@"(?<=\s|\d|^){Regex.Escape(word)}(?=\s|\d|$)", $" {symbol} ", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        }
        text = Regex.Replace(text, @"(?<=\d),(?=\d{3}\b)", ""); // 1,000 is one thousand
        text = text.Replace(',', ' ').Replace("%", "/100", StringComparison.Ordinal);
        try
        {
            var parser = new Parser(text);
            var value = parser.Expression();
            return parser.AtEnd && double.IsFinite(value) ? Format(value) : null;
        }
        catch (FormatException)
        {
            return null;
        }
    }

    private static string Format(double value) =>
        Math.Abs(value - Math.Round(value)) < 1e-9 && Math.Abs(value) < 1e15
            ? Math.Round(value).ToString("N0", CultureInfo.InvariantCulture)
            : value.ToString("#,0.####", CultureInfo.InvariantCulture);

    private static string? Unit(string raw)
    {
        var unit = raw.Trim().ToLowerInvariant();
        if (Units.ContainsKey(unit) || Temperatures.Contains(unit))
        {
            return unit;
        }
        // plurals: "miles", "inches", "feet" is listed, "degrees celsius"
        unit = unit.Replace("degrees ", "", StringComparison.Ordinal).Replace("degree ", "", StringComparison.Ordinal);
        foreach (var candidate in new[] { unit, unit.TrimEnd('s'), unit.EndsWith("es", StringComparison.Ordinal) ? unit[..^2] : unit })
        {
            if (Units.ContainsKey(candidate) || Temperatures.Contains(candidate))
            {
                return candidate;
            }
        }
        return null;
    }

    private static string? Convert(string text)
    {
        var match = ConversionPattern().Match(text);
        if (!match.Success)
        {
            return null;
        }
        var amount = double.Parse(match.Groups[1].Value.Replace(',', '.'), CultureInfo.InvariantCulture);
        var from = Unit(match.Groups[2].Value);
        var to = Unit(match.Groups[3].Value);
        if (from is null || to is null)
        {
            return null;
        }
        if (Temperatures.Contains(from) && Temperatures.Contains(to))
        {
            var kelvin = from[0] switch { 'c' => amount + 273.15, 'f' => (amount - 32) * 5 / 9 + 273.15, _ => amount };
            var result = to[0] switch { 'c' => kelvin - 273.15, 'f' => (kelvin - 273.15) * 9 / 5 + 32, _ => kelvin };
            return $"{Format(result)} °{char.ToUpperInvariant(to[0])}";
        }
        if (!Units.TryGetValue(from, out var source) || !Units.TryGetValue(to, out var target) || source.Dimension != target.Dimension)
        {
            return null;
        }
        return $"{Format(amount * source.Factor / target.Factor)} {match.Groups[3].Value.Trim()}";
    }

    /// <summary>Recursive descent over + - * / ^ and parentheses, with sqrt.</summary>
    private sealed class Parser(string text)
    {
        private int _position;

        public bool AtEnd
        {
            get
            {
                Skip();
                return _position >= text.Length;
            }
        }

        private void Skip()
        {
            while (_position < text.Length && char.IsWhiteSpace(text[_position]))
            {
                _position++;
            }
        }

        private bool Take(char symbol)
        {
            Skip();
            if (_position < text.Length && text[_position] == symbol)
            {
                _position++;
                return true;
            }
            return false;
        }

        public double Expression()
        {
            var value = Term();
            while (true)
            {
                if (Take('+')) value += Term();
                else if (Take('-')) value -= Term();
                else return value;
            }
        }

        private double Term()
        {
            var value = Power();
            while (true)
            {
                if (Take('*')) value *= Power();
                else if (Take('/')) value /= Power();
                else return value;
            }
        }

        private double Power()
        {
            var value = Unary();
            return Take('^') ? Math.Pow(value, Power()) : value;
        }

        private double Unary()
        {
            if (Take('-')) return -Unary();
            if (Take('+')) return Unary();
            Skip();
            if (text.AsSpan(_position).StartsWith("sqrt", StringComparison.OrdinalIgnoreCase))
            {
                _position += 4;
                return Math.Sqrt(Unary());
            }
            if (Take('('))
            {
                var inner = Expression();
                if (!Take(')')) throw new FormatException("missing )");
                return inner;
            }
            var start = _position;
            while (_position < text.Length && (char.IsDigit(text[_position]) || text[_position] == '.'))
            {
                _position++;
            }
            if (start == _position)
            {
                throw new FormatException("number expected");
            }
            return double.Parse(text[start.._position], CultureInfo.InvariantCulture);
        }
    }
}
