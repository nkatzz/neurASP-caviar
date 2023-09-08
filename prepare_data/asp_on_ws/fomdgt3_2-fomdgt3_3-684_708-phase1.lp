
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
orientation(fomdgt3_2,80,700). coords(fomdgt3_2,297,275,700).  visible(fomdgt3_2,700). happensAt(walking(fomdgt3_2),700). orientation(fomdgt3_3,159,700). coords(fomdgt3_3,127,117,700).  visible(fomdgt3_3,700). happensAt(walking(fomdgt3_3),700).
orientation(fomdgt3_2,80,701). coords(fomdgt3_2,297,275,701).  visible(fomdgt3_2,701). happensAt(walking(fomdgt3_2),701). orientation(fomdgt3_3,159,701). coords(fomdgt3_3,127,117,701).  visible(fomdgt3_3,701). happensAt(walking(fomdgt3_3),701).
orientation(fomdgt3_2,80,702). coords(fomdgt3_2,297,276,702).  visible(fomdgt3_2,702). happensAt(walking(fomdgt3_2),702). orientation(fomdgt3_3,159,702). coords(fomdgt3_3,129,118,702).  visible(fomdgt3_3,702). happensAt(walking(fomdgt3_3),702).
orientation(fomdgt3_2,80,703). coords(fomdgt3_2,299,276,703).  visible(fomdgt3_2,703). happensAt(walking(fomdgt3_2),703). orientation(fomdgt3_3,159,703). coords(fomdgt3_3,129,118,703).  visible(fomdgt3_3,703). happensAt(walking(fomdgt3_3),703).
orientation(fomdgt3_2,80,704). coords(fomdgt3_2,299,277,704).  visible(fomdgt3_2,704). happensAt(walking(fomdgt3_2),704). orientation(fomdgt3_3,159,704). coords(fomdgt3_3,129,118,704).  visible(fomdgt3_3,704). happensAt(walking(fomdgt3_3),704).
orientation(fomdgt3_2,80,705). coords(fomdgt3_2,299,278,705).  visible(fomdgt3_2,705). happensAt(walking(fomdgt3_2),705). orientation(fomdgt3_3,159,705). coords(fomdgt3_3,129,118,705).  visible(fomdgt3_3,705). happensAt(walking(fomdgt3_3),705).
orientation(fomdgt3_2,80,706). coords(fomdgt3_2,299,279,706).  visible(fomdgt3_2,706). happensAt(walking(fomdgt3_2),706). orientation(fomdgt3_3,159,706). coords(fomdgt3_3,130,119,706).  visible(fomdgt3_3,706). happensAt(walking(fomdgt3_3),706).
orientation(fomdgt3_2,80,707). coords(fomdgt3_2,302,281,707).  visible(fomdgt3_2,707). happensAt(walking(fomdgt3_2),707). orientation(fomdgt3_3,159,707). coords(fomdgt3_3,130,119,707).  visible(fomdgt3_3,707). happensAt(walking(fomdgt3_3),707).
orientation(fomdgt3_2,80,708). coords(fomdgt3_2,302,283,708).  happensAt(disappear(fomdgt3_2),708). happensAt(walking(fomdgt3_2),708). orientation(fomdgt3_3,159,708). coords(fomdgt3_3,130,119,708).  visible(fomdgt3_3,708). happensAt(walking(fomdgt3_3),708).
#show holdsAt/2.