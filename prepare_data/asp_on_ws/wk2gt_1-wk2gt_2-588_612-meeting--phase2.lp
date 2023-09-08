

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

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,125,588). coords(wk2gt_1,115,53,588).  visible(wk2gt_1,588). orientation(wk2gt_2,142,588). coords(wk2gt_2,99,63,588).  visible(wk2gt_2,588).
orientation(wk2gt_1,125,589). coords(wk2gt_1,115,53,589).  visible(wk2gt_1,589). orientation(wk2gt_2,142,589). coords(wk2gt_2,99,63,589).  visible(wk2gt_2,589).
orientation(wk2gt_1,125,590). coords(wk2gt_1,115,53,590).  visible(wk2gt_1,590). orientation(wk2gt_2,142,590). coords(wk2gt_2,99,63,590).  visible(wk2gt_2,590).
orientation(wk2gt_1,125,591). coords(wk2gt_1,115,53,591).  visible(wk2gt_1,591). orientation(wk2gt_2,142,591). coords(wk2gt_2,99,63,591).  visible(wk2gt_2,591).
orientation(wk2gt_1,125,592). coords(wk2gt_1,115,53,592).  visible(wk2gt_1,592). orientation(wk2gt_2,142,592). coords(wk2gt_2,99,63,592).  visible(wk2gt_2,592).
orientation(wk2gt_1,125,593). coords(wk2gt_1,115,53,593).  visible(wk2gt_1,593). orientation(wk2gt_2,142,593). coords(wk2gt_2,99,63,593).  visible(wk2gt_2,593).
orientation(wk2gt_1,125,594). coords(wk2gt_1,115,53,594).  visible(wk2gt_1,594). orientation(wk2gt_2,142,594). coords(wk2gt_2,99,63,594).  visible(wk2gt_2,594).
orientation(wk2gt_1,125,595). coords(wk2gt_1,115,53,595).  visible(wk2gt_1,595). orientation(wk2gt_2,142,595). coords(wk2gt_2,99,63,595).  visible(wk2gt_2,595).
orientation(wk2gt_1,125,596). coords(wk2gt_1,115,53,596).  visible(wk2gt_1,596). orientation(wk2gt_2,142,596). coords(wk2gt_2,99,63,596).  visible(wk2gt_2,596).
orientation(wk2gt_1,125,597). coords(wk2gt_1,115,53,597).  visible(wk2gt_1,597). orientation(wk2gt_2,142,597). coords(wk2gt_2,99,63,597).  visible(wk2gt_2,597).
orientation(wk2gt_1,125,598). coords(wk2gt_1,115,53,598).  visible(wk2gt_1,598). orientation(wk2gt_2,142,598). coords(wk2gt_2,99,63,598).  visible(wk2gt_2,598).
orientation(wk2gt_1,125,599). coords(wk2gt_1,115,53,599).  visible(wk2gt_1,599). orientation(wk2gt_2,142,599). coords(wk2gt_2,99,63,599).  visible(wk2gt_2,599).
orientation(wk2gt_1,125,600). coords(wk2gt_1,115,53,600).  visible(wk2gt_1,600). orientation(wk2gt_2,142,600). coords(wk2gt_2,99,63,600).  visible(wk2gt_2,600).
orientation(wk2gt_1,125,601). coords(wk2gt_1,115,53,601).  visible(wk2gt_1,601). orientation(wk2gt_2,142,601). coords(wk2gt_2,99,63,601).  visible(wk2gt_2,601).
orientation(wk2gt_1,125,602). coords(wk2gt_1,115,53,602).  visible(wk2gt_1,602). orientation(wk2gt_2,142,602). coords(wk2gt_2,99,63,602).  visible(wk2gt_2,602).
orientation(wk2gt_1,125,603). coords(wk2gt_1,115,53,603).  visible(wk2gt_1,603). orientation(wk2gt_2,142,603). coords(wk2gt_2,99,63,603).  visible(wk2gt_2,603).
orientation(wk2gt_1,125,604). coords(wk2gt_1,115,53,604).  visible(wk2gt_1,604). orientation(wk2gt_2,142,604). coords(wk2gt_2,99,63,604).  visible(wk2gt_2,604).
orientation(wk2gt_1,125,605). coords(wk2gt_1,115,53,605).  visible(wk2gt_1,605). orientation(wk2gt_2,142,605). coords(wk2gt_2,99,63,605).  visible(wk2gt_2,605).
orientation(wk2gt_1,125,606). coords(wk2gt_1,115,53,606).  visible(wk2gt_1,606). orientation(wk2gt_2,142,606). coords(wk2gt_2,99,63,606).  visible(wk2gt_2,606).
orientation(wk2gt_1,125,607). coords(wk2gt_1,115,53,607).  visible(wk2gt_1,607). orientation(wk2gt_2,142,607). coords(wk2gt_2,99,63,607).  visible(wk2gt_2,607).
orientation(wk2gt_1,125,608). coords(wk2gt_1,115,53,608).  visible(wk2gt_1,608). orientation(wk2gt_2,142,608). coords(wk2gt_2,99,63,608).  visible(wk2gt_2,608).
orientation(wk2gt_1,125,609). coords(wk2gt_1,115,53,609).  visible(wk2gt_1,609). orientation(wk2gt_2,142,609). coords(wk2gt_2,99,63,609).  visible(wk2gt_2,609).
orientation(wk2gt_1,125,610). coords(wk2gt_1,114,53,610).  visible(wk2gt_1,610). orientation(wk2gt_2,142,610). coords(wk2gt_2,99,63,610).  visible(wk2gt_2,610).
orientation(wk2gt_1,125,611). coords(wk2gt_1,114,53,611).  visible(wk2gt_1,611). orientation(wk2gt_2,142,611). coords(wk2gt_2,99,63,611).  visible(wk2gt_2,611).
orientation(wk2gt_1,125,612). coords(wk2gt_1,114,53,612).  visible(wk2gt_1,612). orientation(wk2gt_2,142,612). coords(wk2gt_2,99,63,612).  visible(wk2gt_2,612).
orientation(wk2gt_1,125,613). coords(wk2gt_1,114,53,613).  visible(wk2gt_1,613). orientation(wk2gt_2,142,613). coords(wk2gt_2,99,63,613).  visible(wk2gt_2,613).
happensAt(disappear(wk2gt_1),612). happensAt(disappear(wk2gt_2),612).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=588..612.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=588..612.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),588). :- holdsAt(meeting(wk2gt_2,wk2gt_1),588).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),589). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),589).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),590). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),590).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),591). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),591).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),592). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),592).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),593). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),593).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),594). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),594).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),595). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),595).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),596). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),596).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),597). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),597).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),598). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),598).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),599). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),599).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),600). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),600).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),601). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),601).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),602). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),602).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),603). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),603).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),604). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),604).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),605). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),605).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),606). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),606).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),607). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),607).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),608). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),608).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),609). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),609).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),610). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),610).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),611). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),611).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),612). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),612).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),613). :- holdsAt(meeting(wk2gt_2,wk2gt_1),613).
:- simpleEvent(T,P,walking), T>588 , T<612.
:- simpleEvent(T,P,running), T>588 , T<612.

#show simpleEvent/3.