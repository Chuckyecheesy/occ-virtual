"""
test_audio_repeat.py

Unit tests for audio_repeat.py with simulated user input for repeat_mode.
Uses unittest.mock to simulate user input without requiring interactive prompts.
"""

import unittest
from unittest.mock import patch, MagicMock, call
from audio_repeat import read_question_budget_recommendation


class TestAudioRepeatMode(unittest.TestCase):
    """Test cases for repeat_mode functionality in audio_repeat.py"""
    
    def setUp(self):
        """Set up test fixtures."""
        self.question = "What is your current housing situation?"
        self.budget = 1200.50
        self.recommendations = ["Greenwood Apartments", "Sunset Villas", "Maple Residency"]
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_repeat_mode_correct_input_first_try(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test repeat_mode with correct user input on first attempt.
        Expected: Function returns True immediately after correct answer.
        """
        # Mock the pyttsx3 engine
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        # Simulate correct user input on first try
        mock_input.side_effect = [
            "$1,200.50",  # Correct budget format
            "Greenwood Apartments, Sunset Villas, Maple Residency"  # Correct recommendations
        ]
        
        result = read_question_budget_recommendation(
            self.question,
            self.budget,
            self.recommendations,
            repeat_mode=True
        )
        
        self.assertTrue(result)
        self.assertEqual(mock_input.call_count, 2)
        mock_print.assert_any_call("Correct! Well done.")
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_repeat_mode_incorrect_budget_then_correct(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test repeat_mode with incorrect budget first, then correct input.
        Expected: Function retries and returns True on second attempt.
        """
        # Mock the pyttsx3 engine
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        # Simulate incorrect budget first, then both correct
        mock_input.side_effect = [
            "$1,000.00",  # Incorrect budget
            "Greenwood Apartments, Sunset Villas, Maple Residency",  # Correct recommendations (not checked on first fail)
            "$1,200.50",  # Correct budget on retry
            "Greenwood Apartments, Sunset Villas, Maple Residency"  # Correct recommendations on retry
        ]
        
        result = read_question_budget_recommendation(
            self.question,
            self.budget,
            self.recommendations,
            repeat_mode=True
        )
        
        self.assertTrue(result)
        # Should have called input 4 times (2 failed attempts, 2 successful attempts)
        self.assertEqual(mock_input.call_count, 4)
        mock_print.assert_any_call("Incorrect, let's try again.")
        mock_print.assert_any_call("Correct! Well done.")
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_repeat_mode_incorrect_recommendations_then_correct(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test repeat_mode with incorrect recommendations first, then correct input.
        Expected: Function retries after incorrect recommendations.
        """
        # Mock the pyttsx3 engine
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        # Simulate incorrect recommendations first, then correct
        mock_input.side_effect = [
            "$1,200.50",  # Correct budget
            "Greenwood Apartments, Sunset Villas",  # Incorrect recommendations (missing Maple Residency)
            "$1,200.50",  # Correct budget on retry
            "Greenwood Apartments, Sunset Villas, Maple Residency"  # Correct recommendations on retry
        ]
        
        result = read_question_budget_recommendation(
            self.question,
            self.budget,
            self.recommendations,
            repeat_mode=True
        )
        
        self.assertTrue(result)
        self.assertEqual(mock_input.call_count, 4)
        mock_print.assert_any_call("Incorrect, let's try again.")
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_repeat_mode_disabled(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test with repeat_mode=False.
        Expected: Function returns True without requesting user input.
        """
        # Mock the pyttsx3 engine
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        result = read_question_budget_recommendation(
            self.question,
            self.budget,
            self.recommendations,
            repeat_mode=False
        )
        
        self.assertTrue(result)
        # Input should never be called when repeat_mode is False
        mock_input.assert_not_called()
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_repeat_mode_speech_engine_called(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test that pyttsx3 engine is properly initialized and used.
        Expected: Engine speaks the question, budget, and recommendations.
        """
        # Mock the pyttsx3 engine
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        # Simulate correct input on first try
        mock_input.side_effect = [
            "$1,200.50",
            "Greenwood Apartments, Sunset Villas, Maple Residency"
        ]
        
        read_question_budget_recommendation(
            self.question,
            self.budget,
            self.recommendations,
            repeat_mode=True
        )
        
        # Verify engine initialization
        mock_pyttsx3.assert_called_once()
        
        # Verify engine properties were set
        mock_engine.setProperty.assert_called()
        
        # Verify engine spoke the content
        self.assertGreaterEqual(mock_engine.say.call_count, 3)  # At least: question, budget, recommendations
        mock_engine.runAndWait.assert_called()
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_repeat_mode_multiple_retries(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test repeat_mode with multiple incorrect attempts before correct answer.
        Expected: Function retries until correct answer is provided.
        """
        # Mock the pyttsx3 engine
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        # Simulate three failed attempts, then success
        mock_input.side_effect = [
            "$500.00", "Wrong",  # First attempt - both wrong
            "$1,500.00", "Greenwood Apartments, Sunset Villas, Maple Residency",  # Second - budget wrong
            "$1,200.50", "Greenwood Apartments",  # Third - recommendations wrong
            "$1,200.50", "Greenwood Apartments, Sunset Villas, Maple Residency"  # Fourth - correct
        ]
        
        result = read_question_budget_recommendation(
            self.question,
            self.budget,
            self.recommendations,
            repeat_mode=True
        )
        
        self.assertTrue(result)
        self.assertEqual(mock_input.call_count, 8)
        # Should see "Incorrect, let's try again." three times
        self.assertEqual(mock_print.call_args_list.count(call("Incorrect, let's try again.")), 3)


class TestAudioRepeatModeEdgeCases(unittest.TestCase):
    """Test edge cases for repeat_mode functionality."""
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_single_recommendation(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test repeat_mode with a single apartment recommendation.
        Expected: Function handles single recommendation correctly.
        """
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        budget = 800.00
        recommendations = ["Budget Apartments"]
        
        mock_input.side_effect = [
            "$800.00",
            "Budget Apartments"
        ]
        
        result = read_question_budget_recommendation(
            "Choose housing",
            budget,
            recommendations,
            repeat_mode=True
        )
        
        self.assertTrue(result)
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_many_recommendations(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test repeat_mode with many apartment recommendations.
        Expected: Function handles long recommendation lists correctly.
        """
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        budget = 1500.00
        recommendations = [
            "Apt 1", "Apt 2", "Apt 3", "Apt 4", "Apt 5",
            "Apt 6", "Apt 7", "Apt 8", "Apt 9", "Apt 10"
        ]
        
        mock_input.side_effect = [
            "$1,500.00",
            "Apt 1, Apt 2, Apt 3, Apt 4, Apt 5, Apt 6, Apt 7, Apt 8, Apt 9, Apt 10"
        ]
        
        result = read_question_budget_recommendation(
            "Choose housing",
            budget,
            recommendations,
            repeat_mode=True
        )
        
        self.assertTrue(result)
    
    @patch('pyttsx3.init')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_budget_with_decimal_precision(self, mock_print, mock_input, mock_pyttsx3):
        """
        Test repeat_mode with precise decimal budget values.
        Expected: Function correctly formats and validates decimal budgets.
        """
        mock_engine = MagicMock()
        mock_pyttsx3.return_value = mock_engine
        
        budget = 1200.99
        recommendations = ["Apt A", "Apt B"]
        
        mock_input.side_effect = [
            "$1,200.99",
            "Apt A, Apt B"
        ]
        
        result = read_question_budget_recommendation(
            "Choose housing",
            budget,
            recommendations,
            repeat_mode=True
        )
        
        self.assertTrue(result)


def run_tests():
    """Run all tests with verbose output."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
