

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(mwt1gt_5;mwt1gt_6).
orientation(mwt1gt_5,150,682). coords(mwt1gt_5,89,83,682).  visible(mwt1gt_5,682). orientation(mwt1gt_6,150,682). coords(mwt1gt_6,76,93,682).  visible(mwt1gt_6,682).
orientation(mwt1gt_5,150,683). coords(mwt1gt_5,91,83,683).  visible(mwt1gt_5,683). orientation(mwt1gt_6,150,683). coords(mwt1gt_6,77,94,683).  visible(mwt1gt_6,683).
orientation(mwt1gt_5,150,684). coords(mwt1gt_5,91,83,684).  visible(mwt1gt_5,684). orientation(mwt1gt_6,150,684). coords(mwt1gt_6,77,94,684).  visible(mwt1gt_6,684).
orientation(mwt1gt_5,150,685). coords(mwt1gt_5,92,83,685).  visible(mwt1gt_5,685). orientation(mwt1gt_6,150,685). coords(mwt1gt_6,78,95,685).  visible(mwt1gt_6,685).
orientation(mwt1gt_5,150,686). coords(mwt1gt_5,92,83,686).  visible(mwt1gt_5,686). orientation(mwt1gt_6,150,686). coords(mwt1gt_6,78,95,686).  visible(mwt1gt_6,686).
orientation(mwt1gt_5,150,687). coords(mwt1gt_5,92,83,687).  visible(mwt1gt_5,687). orientation(mwt1gt_6,150,687). coords(mwt1gt_6,78,95,687).  visible(mwt1gt_6,687).
orientation(mwt1gt_5,150,688). coords(mwt1gt_5,92,83,688).  visible(mwt1gt_5,688). orientation(mwt1gt_6,150,688). coords(mwt1gt_6,78,95,688).  visible(mwt1gt_6,688).
orientation(mwt1gt_5,150,689). coords(mwt1gt_5,92,83,689).  visible(mwt1gt_5,689). orientation(mwt1gt_6,150,689). coords(mwt1gt_6,78,95,689).  visible(mwt1gt_6,689).
orientation(mwt1gt_5,150,690). coords(mwt1gt_5,95,83,690).  visible(mwt1gt_5,690). orientation(mwt1gt_6,150,690). coords(mwt1gt_6,79,96,690).  visible(mwt1gt_6,690).
orientation(mwt1gt_5,150,691). coords(mwt1gt_5,95,83,691).  visible(mwt1gt_5,691). orientation(mwt1gt_6,150,691). coords(mwt1gt_6,79,96,691).  visible(mwt1gt_6,691).
orientation(mwt1gt_5,150,692). coords(mwt1gt_5,95,83,692).  visible(mwt1gt_5,692). orientation(mwt1gt_6,150,692). coords(mwt1gt_6,79,96,692).  visible(mwt1gt_6,692).
orientation(mwt1gt_5,150,693). coords(mwt1gt_5,95,83,693).  visible(mwt1gt_5,693). orientation(mwt1gt_6,150,693). coords(mwt1gt_6,79,96,693).  visible(mwt1gt_6,693).
orientation(mwt1gt_5,150,694). coords(mwt1gt_5,95,83,694).  visible(mwt1gt_5,694). orientation(mwt1gt_6,150,694). coords(mwt1gt_6,79,96,694).  visible(mwt1gt_6,694).
orientation(mwt1gt_5,150,695). coords(mwt1gt_5,96,83,695).  visible(mwt1gt_5,695). orientation(mwt1gt_6,150,695). coords(mwt1gt_6,80,99,695).  visible(mwt1gt_6,695).
orientation(mwt1gt_5,150,696). coords(mwt1gt_5,96,83,696).  visible(mwt1gt_5,696). orientation(mwt1gt_6,150,696). coords(mwt1gt_6,80,99,696).  visible(mwt1gt_6,696).
orientation(mwt1gt_5,150,697). coords(mwt1gt_5,97,83,697).  visible(mwt1gt_5,697). orientation(mwt1gt_6,150,697). coords(mwt1gt_6,80,99,697).  visible(mwt1gt_6,697).
orientation(mwt1gt_5,150,698). coords(mwt1gt_5,97,83,698).  visible(mwt1gt_5,698). orientation(mwt1gt_6,150,698). coords(mwt1gt_6,80,99,698).  visible(mwt1gt_6,698).
orientation(mwt1gt_5,150,699). coords(mwt1gt_5,98,82,699).  visible(mwt1gt_5,699). orientation(mwt1gt_6,150,699). coords(mwt1gt_6,82,101,699).  visible(mwt1gt_6,699).
orientation(mwt1gt_5,150,700). coords(mwt1gt_5,98,82,700).  visible(mwt1gt_5,700). orientation(mwt1gt_6,150,700). coords(mwt1gt_6,82,101,700).  visible(mwt1gt_6,700).
orientation(mwt1gt_5,150,701). coords(mwt1gt_5,98,82,701).  visible(mwt1gt_5,701). orientation(mwt1gt_6,150,701). coords(mwt1gt_6,82,101,701).  visible(mwt1gt_6,701).
orientation(mwt1gt_5,150,702). coords(mwt1gt_5,101,82,702).  visible(mwt1gt_5,702). orientation(mwt1gt_6,150,702). coords(mwt1gt_6,83,101,702).  visible(mwt1gt_6,702).
orientation(mwt1gt_5,150,703). coords(mwt1gt_5,101,82,703).  visible(mwt1gt_5,703). orientation(mwt1gt_6,150,703). coords(mwt1gt_6,83,101,703).  visible(mwt1gt_6,703).
orientation(mwt1gt_5,150,704). coords(mwt1gt_5,101,82,704).  visible(mwt1gt_5,704). orientation(mwt1gt_6,150,704). coords(mwt1gt_6,83,101,704).  visible(mwt1gt_6,704).
orientation(mwt1gt_5,150,705). coords(mwt1gt_5,101,82,705).  visible(mwt1gt_5,705). orientation(mwt1gt_6,150,705). coords(mwt1gt_6,83,101,705).  visible(mwt1gt_6,705).
orientation(mwt1gt_5,150,706). coords(mwt1gt_5,101,82,706).  happensAt(disappear(mwt1gt_5),706). orientation(mwt1gt_6,150,706). coords(mwt1gt_6,83,101,706).  happensAt(disappear(mwt1gt_6),706).
orientation(mwt1gt_5,150,707). coords(mwt1gt_5,101,82,707).  happensAt(disappear(mwt1gt_5),707). orientation(mwt1gt_6,150,707). coords(mwt1gt_6,83,101,707).  happensAt(disappear(mwt1gt_6),707).
happensAt(disappear(mwt1gt_5),706). happensAt(disappear(mwt1gt_6),706).
1{simpleEvent(T,mwt1gt_5,active); simpleEvent(T,mwt1gt_5,inactive); simpleEvent(T,mwt1gt_5,walking); simpleEvent(T,mwt1gt_5,running)}1 :- T=682..706.
1{simpleEvent(T,mwt1gt_6,active); simpleEvent(T,mwt1gt_6,inactive); simpleEvent(T,mwt1gt_6,walking); simpleEvent(T,mwt1gt_6,running)}1 :- T=682..706.
:- holdsAt(moving(mwt1gt_5,mwt1gt_6),682). :- holdsAt(moving(mwt1gt_6,mwt1gt_5),682).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),683). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),683).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),684). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),684).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),685). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),685).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),686). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),686).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),687). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),687).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),688). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),688).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),689). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),689).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),690). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),690).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),691). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),691).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),692). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),692).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),693). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),693).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),694). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),694).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),695). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),695).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),696). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),696).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),697). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),697).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),698). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),698).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),699). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),699).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),700). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),700).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),701). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),701).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),702). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),702).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),703). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),703).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),704). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),704).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),705). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),705).
:- not holdsAt(moving(mwt1gt_5,mwt1gt_6),706). :- not holdsAt(moving(mwt1gt_6,mwt1gt_5),706).
:- holdsAt(moving(mwt1gt_5,mwt1gt_6),707). :- holdsAt(moving(mwt1gt_6,mwt1gt_5),707).
:- simpleEvent(T,P,inactive), T>682 , T<706.
:- simpleEvent(T,P,running), T>682 , T<706.

#show simpleEvent/3.