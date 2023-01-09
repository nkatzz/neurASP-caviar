
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

	orientation(lb1gt_4,90,1228). coords(lb1gt_4,310,112,1228).  visible(lb1gt_4,1228). happensAt(inactive(lb1gt_4),1228). orientation(lb1gt_5,122,1228). coords(lb1gt_5,174,36,1228).  visible(lb1gt_5,1228). happensAt(walking(lb1gt_5),1228).
orientation(lb1gt_4,90,1229). coords(lb1gt_4,310,112,1229).  visible(lb1gt_4,1229). happensAt(inactive(lb1gt_4),1229). orientation(lb1gt_5,122,1229). coords(lb1gt_5,174,36,1229).  visible(lb1gt_5,1229). happensAt(walking(lb1gt_5),1229).
orientation(lb1gt_4,90,1230). coords(lb1gt_4,310,112,1230).  visible(lb1gt_4,1230). happensAt(inactive(lb1gt_4),1230). orientation(lb1gt_5,125,1230). coords(lb1gt_5,173,37,1230).  visible(lb1gt_5,1230). happensAt(walking(lb1gt_5),1230).
orientation(lb1gt_4,90,1231). coords(lb1gt_4,310,112,1231).  visible(lb1gt_4,1231). happensAt(inactive(lb1gt_4),1231). orientation(lb1gt_5,124,1231). coords(lb1gt_5,173,37,1231).  visible(lb1gt_5,1231). happensAt(walking(lb1gt_5),1231).
orientation(lb1gt_4,90,1232). coords(lb1gt_4,310,112,1232).  visible(lb1gt_4,1232). happensAt(inactive(lb1gt_4),1232). orientation(lb1gt_5,123,1232). coords(lb1gt_5,173,37,1232).  visible(lb1gt_5,1232). happensAt(walking(lb1gt_5),1232).
orientation(lb1gt_4,90,1233). coords(lb1gt_4,310,112,1233).  visible(lb1gt_4,1233). happensAt(inactive(lb1gt_4),1233). orientation(lb1gt_5,124,1233). coords(lb1gt_5,173,37,1233).  visible(lb1gt_5,1233). happensAt(walking(lb1gt_5),1233).
orientation(lb1gt_4,90,1234). coords(lb1gt_4,310,112,1234).  visible(lb1gt_4,1234). happensAt(inactive(lb1gt_4),1234). orientation(lb1gt_5,123,1234). coords(lb1gt_5,172,37,1234).  visible(lb1gt_5,1234). happensAt(walking(lb1gt_5),1234).
orientation(lb1gt_4,90,1235). coords(lb1gt_4,310,112,1235).  visible(lb1gt_4,1235). happensAt(inactive(lb1gt_4),1235). orientation(lb1gt_5,124,1235). coords(lb1gt_5,172,37,1235).  visible(lb1gt_5,1235). happensAt(walking(lb1gt_5),1235).
orientation(lb1gt_4,90,1236). coords(lb1gt_4,310,112,1236).  visible(lb1gt_4,1236). happensAt(inactive(lb1gt_4),1236). orientation(lb1gt_5,127,1236). coords(lb1gt_5,172,38,1236).  visible(lb1gt_5,1236). happensAt(walking(lb1gt_5),1236).
orientation(lb1gt_4,90,1237). coords(lb1gt_4,310,112,1237).  visible(lb1gt_4,1237). happensAt(inactive(lb1gt_4),1237). orientation(lb1gt_5,127,1237). coords(lb1gt_5,173,38,1237).  visible(lb1gt_5,1237). happensAt(walking(lb1gt_5),1237).
orientation(lb1gt_4,90,1238). coords(lb1gt_4,310,112,1238).  visible(lb1gt_4,1238). happensAt(inactive(lb1gt_4),1238). orientation(lb1gt_5,127,1238). coords(lb1gt_5,174,40,1238).  visible(lb1gt_5,1238). happensAt(walking(lb1gt_5),1238).
orientation(lb1gt_4,90,1239). coords(lb1gt_4,310,112,1239).  visible(lb1gt_4,1239). happensAt(inactive(lb1gt_4),1239). orientation(lb1gt_5,127,1239). coords(lb1gt_5,174,41,1239).  visible(lb1gt_5,1239). happensAt(walking(lb1gt_5),1239).
orientation(lb1gt_4,90,1240). coords(lb1gt_4,310,112,1240).  visible(lb1gt_4,1240). happensAt(inactive(lb1gt_4),1240). orientation(lb1gt_5,127,1240). coords(lb1gt_5,174,42,1240).  visible(lb1gt_5,1240). happensAt(walking(lb1gt_5),1240).
orientation(lb1gt_4,90,1241). coords(lb1gt_4,310,112,1241).  visible(lb1gt_4,1241). happensAt(inactive(lb1gt_4),1241). orientation(lb1gt_5,127,1241). coords(lb1gt_5,174,43,1241).  visible(lb1gt_5,1241). happensAt(walking(lb1gt_5),1241).
orientation(lb1gt_4,90,1242). coords(lb1gt_4,310,112,1242).  visible(lb1gt_4,1242). happensAt(inactive(lb1gt_4),1242). orientation(lb1gt_5,127,1242). coords(lb1gt_5,175,44,1242).  visible(lb1gt_5,1242). happensAt(walking(lb1gt_5),1242).
orientation(lb1gt_4,90,1243). coords(lb1gt_4,310,112,1243).  visible(lb1gt_4,1243). happensAt(inactive(lb1gt_4),1243). orientation(lb1gt_5,127,1243). coords(lb1gt_5,175,45,1243).  visible(lb1gt_5,1243). happensAt(walking(lb1gt_5),1243).
orientation(lb1gt_4,90,1244). coords(lb1gt_4,310,112,1244).  visible(lb1gt_4,1244). happensAt(inactive(lb1gt_4),1244). orientation(lb1gt_5,127,1244). coords(lb1gt_5,175,46,1244).  visible(lb1gt_5,1244). happensAt(walking(lb1gt_5),1244).
orientation(lb1gt_4,90,1245). coords(lb1gt_4,310,112,1245).  visible(lb1gt_4,1245). happensAt(inactive(lb1gt_4),1245). orientation(lb1gt_5,127,1245). coords(lb1gt_5,175,46,1245).  visible(lb1gt_5,1245). happensAt(walking(lb1gt_5),1245).
orientation(lb1gt_4,90,1246). coords(lb1gt_4,310,112,1246).  visible(lb1gt_4,1246). happensAt(inactive(lb1gt_4),1246). orientation(lb1gt_5,127,1246). coords(lb1gt_5,175,47,1246).  visible(lb1gt_5,1246). happensAt(walking(lb1gt_5),1246).
orientation(lb1gt_4,90,1247). coords(lb1gt_4,310,112,1247).  visible(lb1gt_4,1247). happensAt(inactive(lb1gt_4),1247). orientation(lb1gt_5,127,1247). coords(lb1gt_5,175,47,1247).  visible(lb1gt_5,1247). happensAt(walking(lb1gt_5),1247).
orientation(lb1gt_4,90,1248). coords(lb1gt_4,310,112,1248).  visible(lb1gt_4,1248). happensAt(inactive(lb1gt_4),1248). orientation(lb1gt_5,127,1248). coords(lb1gt_5,175,47,1248).  visible(lb1gt_5,1248). happensAt(walking(lb1gt_5),1248).
orientation(lb1gt_4,90,1249). coords(lb1gt_4,310,112,1249).  visible(lb1gt_4,1249). happensAt(inactive(lb1gt_4),1249). orientation(lb1gt_5,127,1249). coords(lb1gt_5,175,49,1249).  visible(lb1gt_5,1249). happensAt(walking(lb1gt_5),1249).
orientation(lb1gt_4,90,1250). coords(lb1gt_4,310,112,1250).  visible(lb1gt_4,1250). happensAt(inactive(lb1gt_4),1250). orientation(lb1gt_5,127,1250). coords(lb1gt_5,175,52,1250).  visible(lb1gt_5,1250). happensAt(walking(lb1gt_5),1250).
orientation(lb1gt_4,90,1251). coords(lb1gt_4,310,112,1251).  visible(lb1gt_4,1251). happensAt(inactive(lb1gt_4),1251). orientation(lb1gt_5,127,1251). coords(lb1gt_5,175,53,1251).  visible(lb1gt_5,1251). happensAt(walking(lb1gt_5),1251).
orientation(lb1gt_4,90,1252). coords(lb1gt_4,310,112,1252).  visible(lb1gt_4,1252). happensAt(inactive(lb1gt_4),1252). orientation(lb1gt_5,127,1252). coords(lb1gt_5,175,56,1252).  visible(lb1gt_5,1252). happensAt(walking(lb1gt_5),1252).
#show holdsAt/2.