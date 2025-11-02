from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app=FastAPI(title="IRIS Model APIs")

