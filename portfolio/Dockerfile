FROM node:18-alpine

COPY ./simplefolio /simplefolio
WORKDIR /simplefolio

RUN npm audit fix %% \
    npm i @parcel/transformer-sass && \
    npm install

CMD ["npm", "start"]