
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

	orientation(fomdgt3_2,110,675). coords(fomdgt3_2,294,237,675).  visible(fomdgt3_2,675). happensAt(walking(fomdgt3_2),675). orientation(fomdgt3_3,159,675). coords(fomdgt3_3,103,103,675).  visible(fomdgt3_3,675). happensAt(walking(fomdgt3_3),675).
orientation(fomdgt3_2,110,676). coords(fomdgt3_2,294,241,676).  visible(fomdgt3_2,676). happensAt(walking(fomdgt3_2),676). orientation(fomdgt3_3,159,676). coords(fomdgt3_3,105,103,676).  visible(fomdgt3_3,676). happensAt(walking(fomdgt3_3),676).
orientation(fomdgt3_2,110,677). coords(fomdgt3_2,294,244,677).  visible(fomdgt3_2,677). happensAt(walking(fomdgt3_2),677). orientation(fomdgt3_3,159,677). coords(fomdgt3_3,105,103,677).  visible(fomdgt3_3,677). happensAt(walking(fomdgt3_3),677).
orientation(fomdgt3_2,110,678). coords(fomdgt3_2,294,246,678).  visible(fomdgt3_2,678). happensAt(walking(fomdgt3_2),678). orientation(fomdgt3_3,159,678). coords(fomdgt3_3,106,105,678).  visible(fomdgt3_3,678). happensAt(walking(fomdgt3_3),678).
orientation(fomdgt3_2,110,679). coords(fomdgt3_2,294,246,679).  visible(fomdgt3_2,679). happensAt(walking(fomdgt3_2),679). orientation(fomdgt3_3,159,679). coords(fomdgt3_3,106,105,679).  visible(fomdgt3_3,679). happensAt(walking(fomdgt3_3),679).
orientation(fomdgt3_2,110,680). coords(fomdgt3_2,294,250,680).  visible(fomdgt3_2,680). happensAt(walking(fomdgt3_2),680). orientation(fomdgt3_3,159,680). coords(fomdgt3_3,106,105,680).  visible(fomdgt3_3,680). happensAt(walking(fomdgt3_3),680).
orientation(fomdgt3_2,110,681). coords(fomdgt3_2,294,252,681).  visible(fomdgt3_2,681). happensAt(walking(fomdgt3_2),681). orientation(fomdgt3_3,159,681). coords(fomdgt3_3,106,105,681).  visible(fomdgt3_3,681). happensAt(walking(fomdgt3_3),681).
orientation(fomdgt3_2,110,682). coords(fomdgt3_2,294,253,682).  visible(fomdgt3_2,682). happensAt(walking(fomdgt3_2),682). orientation(fomdgt3_3,159,682). coords(fomdgt3_3,106,105,682).  visible(fomdgt3_3,682). happensAt(walking(fomdgt3_3),682).
orientation(fomdgt3_2,110,683). coords(fomdgt3_2,294,255,683).  visible(fomdgt3_2,683). happensAt(walking(fomdgt3_2),683). orientation(fomdgt3_3,159,683). coords(fomdgt3_3,107,105,683).  visible(fomdgt3_3,683). happensAt(walking(fomdgt3_3),683).
orientation(fomdgt3_2,110,684). coords(fomdgt3_2,294,256,684).  visible(fomdgt3_2,684). happensAt(walking(fomdgt3_2),684). orientation(fomdgt3_3,159,684). coords(fomdgt3_3,109,105,684).  visible(fomdgt3_3,684). happensAt(walking(fomdgt3_3),684).
orientation(fomdgt3_2,110,685). coords(fomdgt3_2,294,257,685).  visible(fomdgt3_2,685). happensAt(walking(fomdgt3_2),685). orientation(fomdgt3_3,159,685). coords(fomdgt3_3,109,105,685).  visible(fomdgt3_3,685). happensAt(walking(fomdgt3_3),685).
orientation(fomdgt3_2,80,686). coords(fomdgt3_2,294,258,686).  visible(fomdgt3_2,686). happensAt(walking(fomdgt3_2),686). orientation(fomdgt3_3,159,686). coords(fomdgt3_3,113,106,686).  visible(fomdgt3_3,686). happensAt(walking(fomdgt3_3),686).
orientation(fomdgt3_2,80,687). coords(fomdgt3_2,294,259,687).  visible(fomdgt3_2,687). happensAt(walking(fomdgt3_2),687). orientation(fomdgt3_3,159,687). coords(fomdgt3_3,114,107,687).  visible(fomdgt3_3,687). happensAt(walking(fomdgt3_3),687).
orientation(fomdgt3_2,80,688). coords(fomdgt3_2,294,261,688).  visible(fomdgt3_2,688). happensAt(walking(fomdgt3_2),688). orientation(fomdgt3_3,159,688). coords(fomdgt3_3,114,107,688).  visible(fomdgt3_3,688). happensAt(walking(fomdgt3_3),688).
orientation(fomdgt3_2,80,689). coords(fomdgt3_2,297,264,689).  visible(fomdgt3_2,689). happensAt(walking(fomdgt3_2),689). orientation(fomdgt3_3,159,689). coords(fomdgt3_3,118,108,689).  visible(fomdgt3_3,689). happensAt(walking(fomdgt3_3),689).
orientation(fomdgt3_2,80,690). coords(fomdgt3_2,298,267,690).  visible(fomdgt3_2,690). happensAt(walking(fomdgt3_2),690). orientation(fomdgt3_3,159,690). coords(fomdgt3_3,118,108,690).  visible(fomdgt3_3,690). happensAt(walking(fomdgt3_3),690).
orientation(fomdgt3_2,80,691). coords(fomdgt3_2,297,270,691).  visible(fomdgt3_2,691). happensAt(walking(fomdgt3_2),691). orientation(fomdgt3_3,159,691). coords(fomdgt3_3,118,108,691).  visible(fomdgt3_3,691). happensAt(walking(fomdgt3_3),691).
orientation(fomdgt3_2,80,692). coords(fomdgt3_2,297,272,692).  visible(fomdgt3_2,692). happensAt(walking(fomdgt3_2),692). orientation(fomdgt3_3,159,692). coords(fomdgt3_3,118,108,692).  visible(fomdgt3_3,692). happensAt(walking(fomdgt3_3),692).
orientation(fomdgt3_2,80,693). coords(fomdgt3_2,297,273,693).  visible(fomdgt3_2,693). happensAt(walking(fomdgt3_2),693). orientation(fomdgt3_3,159,693). coords(fomdgt3_3,119,109,693).  visible(fomdgt3_3,693). happensAt(walking(fomdgt3_3),693).
orientation(fomdgt3_2,80,694). coords(fomdgt3_2,297,275,694).  visible(fomdgt3_2,694). happensAt(walking(fomdgt3_2),694). orientation(fomdgt3_3,159,694). coords(fomdgt3_3,119,109,694).  visible(fomdgt3_3,694). happensAt(walking(fomdgt3_3),694).
orientation(fomdgt3_2,80,695). coords(fomdgt3_2,297,276,695).  visible(fomdgt3_2,695). happensAt(walking(fomdgt3_2),695). orientation(fomdgt3_3,159,695). coords(fomdgt3_3,120,110,695).  visible(fomdgt3_3,695). happensAt(walking(fomdgt3_3),695).
orientation(fomdgt3_2,80,696). coords(fomdgt3_2,297,276,696).  visible(fomdgt3_2,696). happensAt(walking(fomdgt3_2),696). orientation(fomdgt3_3,159,696). coords(fomdgt3_3,120,110,696).  visible(fomdgt3_3,696). happensAt(walking(fomdgt3_3),696).
orientation(fomdgt3_2,80,697). coords(fomdgt3_2,297,277,697).  visible(fomdgt3_2,697). happensAt(walking(fomdgt3_2),697). orientation(fomdgt3_3,159,697). coords(fomdgt3_3,122,112,697).  visible(fomdgt3_3,697). happensAt(walking(fomdgt3_3),697).
orientation(fomdgt3_2,80,698). coords(fomdgt3_2,297,277,698).  visible(fomdgt3_2,698). happensAt(walking(fomdgt3_2),698). orientation(fomdgt3_3,159,698). coords(fomdgt3_3,124,114,698).  visible(fomdgt3_3,698). happensAt(walking(fomdgt3_3),698).
orientation(fomdgt3_2,80,699). coords(fomdgt3_2,297,275,699).  visible(fomdgt3_2,699). happensAt(walking(fomdgt3_2),699). orientation(fomdgt3_3,159,699). coords(fomdgt3_3,126,116,699).  visible(fomdgt3_3,699). happensAt(walking(fomdgt3_3),699).
#show holdsAt/2.