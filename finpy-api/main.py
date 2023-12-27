from fastapi import FastAPI, Path, Query, Body
from fastapi import File, UploadFile
from pydantic import BaseModel, HttpUrl

from enum import Enum

import pandas as pd
import numpy as np

from alpha_vantage.fundamentaldata import FundamentalData
fd = FundamentalData(key='CMDH1XIYJQRHDGJT', output_format='pandas', indexing_type='date')

from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='CMDH1XIYJQRHDGJT', output_format='pandas', indexing_type='date')


# Calculator
from functions.calculator import calculate

# Machine Learning Model - Style Transfer
import time
import uuid
import cv2
import uvicorn
from PIL import Image

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import ml_models.style_transfer.config as config
import ml_models.style_transfer.inference as inference

# Machine Learning Model - Bank Notes
from ml_models.bank_notes.BankNotes import BankNote
import pickle

# Machine Learning Model - Insurance
import io
import h2o

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from ml_models.health_insurance.utils.data_processing import match_col_types, separate_id_col

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/add/{number}")
async def add_number(number: int):
    big_number = number + 5
    return {"big_number": big_number}


# Stocks
@app.get("/stock/{ticker}")
async def get_stock(ticker: str):
    data = ts.get_intraday(str(ticker))
    price = data
    # price = data["1. open"][0] # type: ignore
    return {"price": price}


# Machine Learning
class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


# Optional parameters
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}


### Request Body
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    item.price = item.price + 10
    item.name.capitalize
    return item

# Use the model
class ItemTax(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/tax/")
async def tax_item(item: ItemTax):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    item_dict.update({"extra_parameter": "roy_check"})
    return item_dict


# Request body + path + query parameters
class Item2(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

# Request body + path + query parameters
@app.put("/tax/{item_id}") # http://127.0.0.1:8000/tax/3?q=alabala
async def create_item_solo(item_id: int, item: Item2, q: str | None = None): 
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result



### Query Parameters and String Validations

# Additional validation
@app.get("/additional_validation/")
# Enforce that even though q is optional, whenever it is provided, its length doesn't exceed 50 characters.
async def additional_validation(q: str | None = Query(default=None, max_length=50)):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q}) # type: ignore
    return results


# Add regular expressions
@app.get("/regular_expressions/")
async def regular_expressions(
    q: str
    | None = Query(default=None, min_length=3, max_length=50, regex="^fixedquery$")
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q}) # type: ignore
    return results

# Query parameter list / multiple values with defaults
@app.get("/multiple_values/") # http://localhost:8000/items/?q=foo&q=bar
# You can also use list directly instead of List[str] (or list[str] in Python 3.9+).
async def multiple_values(q: list[str] | None = Query(default=None)): # default=["foo", "bar"]; 
    query_items = {"q": q}
    return query_items

# Deprecating parameters
@app.get("/deprecating_parameters/")
async def deprecating_parameters(
    q: str
    | None = Query(
        default=None,
        alias="item-query",
        title="Query string",
        description="Query string for the items to search in the database that have a good match",
        min_length=3,
        max_length=50,
        regex="^fixedquery$",
        deprecated=True,
        # include_in_schema=False,
    )
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q}) # type: ignore
    return results


### Path Parameters and Numeric Validations

# Declare metadata
@app.get("/declare_metadata/{item_id}")
async def declare_metadata(
    # You can declare the same type of validations and metadata for path parameters with Path.
    item_id: int = Path(title="The ID of the item to get"),
    q: str | None = Query(default=None, alias="item-query"),
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q}) # type: ignore
    return results

# Number validations: floats, greater than and less than
@app.get("/number_validations/{item_id}")
async def number_validations(
    # Python won't do anything with that *, but it will know that all the following parameters
    # should be called as keyword arguments (key-value pairs), also known as kwargs.
    *,
    item_id: int = Path(title="The ID of the item to get", ge=0, le=1000),
    q: str,
    size: float = Query(gt=0, lt=10.5)
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q}) # type: ignore
    return results

### Body - Multiple Parameters

# Multiple body parameters
class ItemMBP(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class UserMBP(BaseModel):
    username: str
    full_name: str | None = None


@app.put("/items/{item_id}")
async def update_item_MBP(item_id: int, item: ItemMBP, user: UserMBP):
    results = {"item_id": item_id, "item": item, "user": user}
    return results

JSON_body_expected = {
    "item": {
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    },
    "user": {
        "username": "dave",
        "full_name": "Dave Grohl"
    }
}

# Multiple body params and query
@app.put("/items_MBPQ/{item_id}")
async def update_item_MBPQ(
    *,
    item_id: int,
    item: ItemMBP = Body(embed=True), # if you want it to expect a JSON with a key "item"
    user: UserMBP,
    importance: int = Body(gt=0),
    q: str | None = None
):
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    if q:
        results.update({"q": q})
    return results


### Body - Nested Models

# Declare a list with a type parameter
class ItemDLTP(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: list[str] = [] # basic
    # With this, even if you receive a request with duplicate data,
    # it will be converted to a set of unique items.
    tags_different: set[str] = set()


@app.put("/items_DLTP/{item_id}")
async def update_item_DLTP(item_id: int, item: ItemDLTP):
    results = {"item_id": item_id, "item": item}
    return results

# Attributes with lists of submodels
class Image_ALS(BaseModel):
    url: HttpUrl
    name: str


class Item_ALS(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = set()
    images: list[Image_ALS] | None = None


@app.put("/items_ALS/{item_id}")
async def update_item_ALS(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results

JSON_body_expected = {
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2,
    "tags": [
        "rock",
        "metal",
        "bar"
    ],
    "images": [
        {
            "url": "http://example.com/baz.jpg",
            "name": "The Foo live"
        },
        {
            "url": "http://example.com/dave.jpg",
            "name": "The Baz"
        }
    ]
}

# Bodies of arbitrary dicts
@app.post("/index-weights/")
async def create_index_weights(weights: dict[int, float]):
    return weights

# Calculator
class User_input(BaseModel):
    operation: str
    x: float
    y: float

@app.post("/calculate/")
def operate(input: User_input):
    result = calculate(input.operation, input.x, input.y)
    return result

# Machine Learning Model - Style Transfer
@app.post("/ml_models/style_transfer/{style}")
# def get_image(style: str, file: UploadFile = File(...)):
#     image = np.array(Image.open(file.file))
#     model = config.STYLES[style]
#     output, resized = inference.inference(model, image)
#     name = f"/storage/{str(uuid.uuid4())}.jpg"
#     cv2.imwrite(name, output)
#     return {"name": name}

async def generate_remaining_models(models, image, name: str):
    executor = ProcessPoolExecutor()
    event_loop = asyncio.get_event_loop()
    await event_loop.run_in_executor(
        executor, partial(process_image, models, image, name)
    )

def process_image(models, image, name: str):
    for model in models:
        output, resized = inference.inference(models[model], image)
        name = name.split(".")[0]
        name = f"{name.split('_')[0]}_{models[model]}.jpg"
        cv2.imwrite(name, output)


@app.post("/ml_models/style_transfer/{style}")
async def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    model = config.STYLES[style]
    start = time.time()
    output, resized = inference.inference(model, image)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    cv2.imwrite(name, output)
    models = config.STYLES.copy()
    del models[style]
    asyncio.create_task(generate_remaining_models(models, image, name))
    return {"name": name, "time": time.time() - start}

# Machine Learning Model - Bank Notes
pickle_in = open("ml_models/bank_notes/classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.post('/ml_models/bank_notes/predict')
def predict_banknote(data: BankNote):
    data_dict = data.dict()
    variance = data_dict['variance']
    skewness = data_dict['skewness']
    kurtosis = data_dict['kurtosis']
    entropy = data_dict['entropy']
    prediction = classifier.predict([[variance, skewness, kurtosis, entropy]])

    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {'prediction': prediction}
    # return prediction

# Machine Learning Model - Insurance
# Initiate H2O instance and MLflow client
# h2o.init()
# client = MlflowClient()

# # Load best model (based on logloss) amongst all experiment runs
# all_exps = [exp.experiment_id for exp in client.list_experiments()]
# runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
# run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmin()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmin()]['experiment_id']
# print(f'Loading best model: Run {run_id} of Experiment {exp_id}')
# best_model = mlflow.h2o.load_model(f"ml_models/health_insurance/mlruns/{exp_id}/{run_id}/artifacts/model/")

# # Create POST endpoint with path '/predict'
# @app.post("/ml_models/health_insurance/predict")
# async def predict(file: bytes = File(...)):
#     print('[+] Initiate Prediction')
#     file_obj = io.BytesIO(file)
#     test_df = pd.read_csv(file_obj)
#     test_h2o = h2o.H2OFrame(test_df)

#     # Separate ID column (if any)
#     id_name, X_id, X_h2o = separate_id_col(test_h2o)

#     # Match test set column types with train set
#     X_h2o = match_col_types(X_h2o)

#     # Generate predictions with best model (output is H2O frame)
#     preds = best_model.predict(X_h2o)
    
#     # Apply processing if dataset has ID column
#     if id_name is not None:
#         preds_list = preds.as_data_frame()['predict'].tolist()
#         id_list = X_id.as_data_frame()[id_name].tolist()
#         preds_final = dict(zip(id_list, preds_list))
#     else:
#         preds_final = preds.as_data_frame()['predict'].tolist()

#     # Convert predictions into JSON format
#     json_compatible_item_data = jsonable_encoder(preds_final)
#     return JSONResponse(content=json_compatible_item_data)

# @app.get("/ml_models/health_insurance/")
# async def main():
#     content = """
#     <body>
#     <h2> Welcome to the End to End AutoML Pipeline Project for Insurance Cross-Sell</h2>
#     <p> The H2O model and FastAPI instances have been set up successfully </p>
#     <p> You can view the FastAPI UI by heading to localhost:8000 </p>
#     <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
#     </body>
#     """
#     # content = """
#     # <body>
#     # <form action="/predict/" enctype="multipart/form-data" method="post">
#     # <input name="file" type="file" multiple>
#     # <input type="submit">
#     # </form>
#     # </body>
#     # """
#     return HTMLResponse(content=content)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)