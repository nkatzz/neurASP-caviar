

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
orientation(wk2gt_1,125,744). coords(wk2gt_1,114,53,744).  visible(wk2gt_1,744). orientation(wk2gt_2,142,744). coords(wk2gt_2,99,63,744).  visible(wk2gt_2,744).
orientation(wk2gt_1,125,745). coords(wk2gt_1,115,53,745).  visible(wk2gt_1,745). orientation(wk2gt_2,142,745). coords(wk2gt_2,99,63,745).  visible(wk2gt_2,745).
orientation(wk2gt_1,125,746). coords(wk2gt_1,115,53,746).  visible(wk2gt_1,746). orientation(wk2gt_2,142,746). coords(wk2gt_2,99,63,746).  visible(wk2gt_2,746).
orientation(wk2gt_1,125,747). coords(wk2gt_1,115,53,747).  visible(wk2gt_1,747). orientation(wk2gt_2,142,747). coords(wk2gt_2,99,63,747).  visible(wk2gt_2,747).
orientation(wk2gt_1,125,748). coords(wk2gt_1,115,53,748).  visible(wk2gt_1,748). orientation(wk2gt_2,142,748). coords(wk2gt_2,99,63,748).  visible(wk2gt_2,748).
orientation(wk2gt_1,125,749). coords(wk2gt_1,115,53,749).  visible(wk2gt_1,749). orientation(wk2gt_2,142,749). coords(wk2gt_2,99,63,749).  visible(wk2gt_2,749).
orientation(wk2gt_1,125,750). coords(wk2gt_1,115,53,750).  visible(wk2gt_1,750). orientation(wk2gt_2,142,750). coords(wk2gt_2,99,63,750).  visible(wk2gt_2,750).
orientation(wk2gt_1,125,751). coords(wk2gt_1,115,53,751).  visible(wk2gt_1,751). orientation(wk2gt_2,142,751). coords(wk2gt_2,99,63,751).  visible(wk2gt_2,751).
orientation(wk2gt_1,125,752). coords(wk2gt_1,115,53,752).  visible(wk2gt_1,752). orientation(wk2gt_2,142,752). coords(wk2gt_2,99,63,752).  visible(wk2gt_2,752).
orientation(wk2gt_1,125,753). coords(wk2gt_1,115,53,753).  visible(wk2gt_1,753). orientation(wk2gt_2,142,753). coords(wk2gt_2,99,63,753).  visible(wk2gt_2,753).
orientation(wk2gt_1,125,754). coords(wk2gt_1,115,53,754).  visible(wk2gt_1,754). orientation(wk2gt_2,142,754). coords(wk2gt_2,99,63,754).  visible(wk2gt_2,754).
orientation(wk2gt_1,125,755). coords(wk2gt_1,115,53,755).  visible(wk2gt_1,755). orientation(wk2gt_2,142,755). coords(wk2gt_2,99,63,755).  visible(wk2gt_2,755).
orientation(wk2gt_1,125,756). coords(wk2gt_1,115,53,756).  visible(wk2gt_1,756). orientation(wk2gt_2,142,756). coords(wk2gt_2,99,63,756).  visible(wk2gt_2,756).
orientation(wk2gt_1,125,757). coords(wk2gt_1,115,53,757).  visible(wk2gt_1,757). orientation(wk2gt_2,142,757). coords(wk2gt_2,99,63,757).  visible(wk2gt_2,757).
orientation(wk2gt_1,125,758). coords(wk2gt_1,115,53,758).  visible(wk2gt_1,758). orientation(wk2gt_2,142,758). coords(wk2gt_2,99,63,758).  visible(wk2gt_2,758).
orientation(wk2gt_1,125,759). coords(wk2gt_1,115,53,759).  visible(wk2gt_1,759). orientation(wk2gt_2,142,759). coords(wk2gt_2,99,63,759).  visible(wk2gt_2,759).
orientation(wk2gt_1,125,760). coords(wk2gt_1,115,53,760).  visible(wk2gt_1,760). orientation(wk2gt_2,142,760). coords(wk2gt_2,99,63,760).  visible(wk2gt_2,760).
orientation(wk2gt_1,125,761). coords(wk2gt_1,115,53,761).  visible(wk2gt_1,761). orientation(wk2gt_2,142,761). coords(wk2gt_2,99,63,761).  visible(wk2gt_2,761).
orientation(wk2gt_1,125,762). coords(wk2gt_1,115,53,762).  visible(wk2gt_1,762). orientation(wk2gt_2,142,762). coords(wk2gt_2,99,63,762).  visible(wk2gt_2,762).
orientation(wk2gt_1,125,763). coords(wk2gt_1,115,53,763).  visible(wk2gt_1,763). orientation(wk2gt_2,142,763). coords(wk2gt_2,99,63,763).  visible(wk2gt_2,763).
orientation(wk2gt_1,125,764). coords(wk2gt_1,115,53,764).  visible(wk2gt_1,764). orientation(wk2gt_2,142,764). coords(wk2gt_2,99,63,764).  visible(wk2gt_2,764).
orientation(wk2gt_1,125,765). coords(wk2gt_1,115,53,765).  visible(wk2gt_1,765). orientation(wk2gt_2,142,765). coords(wk2gt_2,99,63,765).  visible(wk2gt_2,765).
orientation(wk2gt_1,125,766). coords(wk2gt_1,115,53,766).  visible(wk2gt_1,766). orientation(wk2gt_2,142,766). coords(wk2gt_2,99,63,766).  visible(wk2gt_2,766).
orientation(wk2gt_1,125,767). coords(wk2gt_1,115,53,767).  visible(wk2gt_1,767). orientation(wk2gt_2,142,767). coords(wk2gt_2,99,63,767).  visible(wk2gt_2,767).
orientation(wk2gt_1,125,768). coords(wk2gt_1,115,53,768).  visible(wk2gt_1,768). orientation(wk2gt_2,142,768). coords(wk2gt_2,99,63,768).  visible(wk2gt_2,768).
orientation(wk2gt_1,125,769). coords(wk2gt_1,115,53,769).  visible(wk2gt_1,769). orientation(wk2gt_2,142,769). coords(wk2gt_2,99,63,769).  visible(wk2gt_2,769).
happensAt(disappear(wk2gt_1),768). happensAt(disappear(wk2gt_2),768).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=744..768.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=744..768.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),744). :- holdsAt(meeting(wk2gt_2,wk2gt_1),744).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),745). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),745).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),746). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),746).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),747). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),747).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),748). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),748).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),749). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),749).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),750). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),750).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),751). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),751).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),752). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),752).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),753). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),753).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),754). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),754).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),755). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),755).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),756). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),756).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),757). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),757).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),758). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),758).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),759). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),759).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),760). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),760).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),761). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),761).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),762). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),762).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),763). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),763).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),764). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),764).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),765). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),765).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),766). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),766).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),767). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),767).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),768). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),768).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),769). :- holdsAt(meeting(wk2gt_2,wk2gt_1),769).
:- simpleEvent(T,P,walking), T>744 , T<768.
:- simpleEvent(T,P,running), T>744 , T<768.

#show simpleEvent/3.