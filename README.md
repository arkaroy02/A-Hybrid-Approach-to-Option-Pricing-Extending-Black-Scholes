# Hybrid Option Pricing Model

## Overview

This project focuses on improving classical option pricing techniques by combining the Black-Scholes model with data-driven enhancements. The goal is to bridge the gap between theoretical pricing and real-world market behavior using empirical features and machine learning corrections.

## Motivation

Traditional models like Black-Scholes assume constant volatility and ideal market conditions, which often leads to mispricing in practice. This project aims to address these limitations by introducing a hybrid framework that incorporates historical volatility, market indicators, and learned corrections.

## Methodology

* Collected options and underlying data (2023–2025)
* Performed data cleaning and feature engineering
* Computed realized volatility using log returns
* Implemented Black-Scholes as a baseline model
* Developed hybrid models with additional market features
* Introduced machine learning-based correction layers

## Current Progress

* Data pipeline and preprocessing completed
* Baseline and hybrid pricing models implemented
* Initial results show improved pricing accuracy over Black-Scholes

## Work in Progress 🚧

This is an ongoing project. Future work includes:

* Expanding data collection and improving data quality
* Extensive backtesting across different market conditions
* Incorporating exogenous variables (macro indicators, etc.)
* Exploring advanced modeling techniques (time-series models, etc.)
* Improving model robustness and generalization

## Tech Stack

* Python (NumPy, Pandas, SciPy)
* Machine Learning libraries (Scikit-learn / TensorFlow / PyTorch)
* Financial modeling and statistical analysis

## Goal

To build a robust, interpretable, and scalable option pricing framework that performs reliably under real market conditions.

---
