
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lb1gt_4,90,1276). coords(lb1gt_4,310,112,1276).  visible(lb1gt_4,1276). happensAt(inactive(lb1gt_4),1276). orientation(lb1gt_5,133,1276). coords(lb1gt_5,189,87,1276).  visible(lb1gt_5,1276). happensAt(walking(lb1gt_5),1276).
orientation(lb1gt_4,90,1277). coords(lb1gt_4,310,112,1277).  visible(lb1gt_4,1277). happensAt(inactive(lb1gt_4),1277). orientation(lb1gt_5,133,1277). coords(lb1gt_5,189,90,1277).  visible(lb1gt_5,1277). happensAt(walking(lb1gt_5),1277).
orientation(lb1gt_4,90,1278). coords(lb1gt_4,310,112,1278).  visible(lb1gt_4,1278). happensAt(inactive(lb1gt_4),1278). orientation(lb1gt_5,133,1278). coords(lb1gt_5,189,92,1278).  visible(lb1gt_5,1278). happensAt(walking(lb1gt_5),1278).
orientation(lb1gt_4,90,1279). coords(lb1gt_4,310,112,1279).  visible(lb1gt_4,1279). happensAt(inactive(lb1gt_4),1279). orientation(lb1gt_5,133,1279). coords(lb1gt_5,190,94,1279).  visible(lb1gt_5,1279). happensAt(walking(lb1gt_5),1279).
orientation(lb1gt_4,90,1280). coords(lb1gt_4,310,112,1280).  visible(lb1gt_4,1280). happensAt(inactive(lb1gt_4),1280). orientation(lb1gt_5,133,1280). coords(lb1gt_5,190,95,1280).  visible(lb1gt_5,1280). happensAt(walking(lb1gt_5),1280).
orientation(lb1gt_4,90,1281). coords(lb1gt_4,310,112,1281).  visible(lb1gt_4,1281). happensAt(inactive(lb1gt_4),1281). orientation(lb1gt_5,133,1281). coords(lb1gt_5,191,95,1281).  visible(lb1gt_5,1281). happensAt(walking(lb1gt_5),1281).
orientation(lb1gt_4,90,1282). coords(lb1gt_4,310,112,1282).  visible(lb1gt_4,1282). happensAt(inactive(lb1gt_4),1282). orientation(lb1gt_5,133,1282). coords(lb1gt_5,192,96,1282).  visible(lb1gt_5,1282). happensAt(walking(lb1gt_5),1282).
orientation(lb1gt_4,90,1283). coords(lb1gt_4,310,112,1283).  visible(lb1gt_4,1283). happensAt(inactive(lb1gt_4),1283). orientation(lb1gt_5,133,1283). coords(lb1gt_5,194,96,1283).  visible(lb1gt_5,1283). happensAt(walking(lb1gt_5),1283).
orientation(lb1gt_4,90,1284). coords(lb1gt_4,310,112,1284).  visible(lb1gt_4,1284). happensAt(inactive(lb1gt_4),1284). orientation(lb1gt_5,133,1284). coords(lb1gt_5,195,96,1284).  visible(lb1gt_5,1284). happensAt(walking(lb1gt_5),1284).
orientation(lb1gt_4,90,1285). coords(lb1gt_4,310,112,1285).  visible(lb1gt_4,1285). happensAt(inactive(lb1gt_4),1285). orientation(lb1gt_5,133,1285). coords(lb1gt_5,198,96,1285).  visible(lb1gt_5,1285). happensAt(walking(lb1gt_5),1285).
orientation(lb1gt_4,90,1286). coords(lb1gt_4,310,112,1286).  visible(lb1gt_4,1286). happensAt(inactive(lb1gt_4),1286). orientation(lb1gt_5,133,1286). coords(lb1gt_5,201,97,1286).  visible(lb1gt_5,1286). happensAt(walking(lb1gt_5),1286).
orientation(lb1gt_4,90,1287). coords(lb1gt_4,310,112,1287).  visible(lb1gt_4,1287). happensAt(inactive(lb1gt_4),1287). orientation(lb1gt_5,133,1287). coords(lb1gt_5,204,97,1287).  visible(lb1gt_5,1287). happensAt(walking(lb1gt_5),1287).
orientation(lb1gt_4,90,1288). coords(lb1gt_4,310,112,1288).  visible(lb1gt_4,1288). happensAt(inactive(lb1gt_4),1288). orientation(lb1gt_5,133,1288). coords(lb1gt_5,207,97,1288).  visible(lb1gt_5,1288). happensAt(walking(lb1gt_5),1288).
orientation(lb1gt_4,90,1289). coords(lb1gt_4,310,112,1289).  visible(lb1gt_4,1289). happensAt(inactive(lb1gt_4),1289). orientation(lb1gt_5,133,1289). coords(lb1gt_5,209,98,1289).  visible(lb1gt_5,1289). happensAt(walking(lb1gt_5),1289).
orientation(lb1gt_4,90,1290). coords(lb1gt_4,310,112,1290).  visible(lb1gt_4,1290). happensAt(inactive(lb1gt_4),1290). orientation(lb1gt_5,133,1290). coords(lb1gt_5,211,98,1290).  visible(lb1gt_5,1290). happensAt(walking(lb1gt_5),1290).
orientation(lb1gt_4,90,1291). coords(lb1gt_4,310,112,1291).  visible(lb1gt_4,1291). happensAt(inactive(lb1gt_4),1291). orientation(lb1gt_5,133,1291). coords(lb1gt_5,213,98,1291).  visible(lb1gt_5,1291). happensAt(walking(lb1gt_5),1291).
orientation(lb1gt_4,90,1292). coords(lb1gt_4,310,112,1292).  visible(lb1gt_4,1292). happensAt(inactive(lb1gt_4),1292). orientation(lb1gt_5,133,1292). coords(lb1gt_5,213,98,1292).  visible(lb1gt_5,1292). happensAt(walking(lb1gt_5),1292).
orientation(lb1gt_4,90,1293). coords(lb1gt_4,310,112,1293).  visible(lb1gt_4,1293). happensAt(inactive(lb1gt_4),1293). orientation(lb1gt_5,133,1293). coords(lb1gt_5,214,98,1293).  visible(lb1gt_5,1293). happensAt(walking(lb1gt_5),1293).
orientation(lb1gt_4,90,1294). coords(lb1gt_4,310,112,1294).  visible(lb1gt_4,1294). happensAt(inactive(lb1gt_4),1294). orientation(lb1gt_5,133,1294). coords(lb1gt_5,216,98,1294).  visible(lb1gt_5,1294). happensAt(walking(lb1gt_5),1294).
orientation(lb1gt_4,90,1295). coords(lb1gt_4,310,112,1295).  visible(lb1gt_4,1295). happensAt(inactive(lb1gt_4),1295). orientation(lb1gt_5,133,1295). coords(lb1gt_5,217,99,1295).  visible(lb1gt_5,1295). happensAt(walking(lb1gt_5),1295).
orientation(lb1gt_4,90,1296). coords(lb1gt_4,310,112,1296).  visible(lb1gt_4,1296). happensAt(inactive(lb1gt_4),1296). orientation(lb1gt_5,133,1296). coords(lb1gt_5,218,100,1296).  visible(lb1gt_5,1296). happensAt(walking(lb1gt_5),1296).
orientation(lb1gt_4,90,1297). coords(lb1gt_4,310,112,1297).  visible(lb1gt_4,1297). happensAt(inactive(lb1gt_4),1297). orientation(lb1gt_5,133,1297). coords(lb1gt_5,219,100,1297).  visible(lb1gt_5,1297). happensAt(walking(lb1gt_5),1297).
orientation(lb1gt_4,90,1298). coords(lb1gt_4,310,112,1298).  visible(lb1gt_4,1298). happensAt(inactive(lb1gt_4),1298). orientation(lb1gt_5,133,1298). coords(lb1gt_5,221,100,1298).  visible(lb1gt_5,1298). happensAt(walking(lb1gt_5),1298).
orientation(lb1gt_4,90,1299). coords(lb1gt_4,310,112,1299).  visible(lb1gt_4,1299). happensAt(inactive(lb1gt_4),1299). orientation(lb1gt_5,133,1299). coords(lb1gt_5,226,99,1299).  visible(lb1gt_5,1299). happensAt(walking(lb1gt_5),1299).
orientation(lb1gt_4,90,1300). coords(lb1gt_4,310,112,1300).  visible(lb1gt_4,1300). happensAt(inactive(lb1gt_4),1300). orientation(lb1gt_5,133,1300). coords(lb1gt_5,231,99,1300).  visible(lb1gt_5,1300). happensAt(walking(lb1gt_5),1300).
#show holdsAt/2.