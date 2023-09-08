

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
orientation(wk2gt_1,125,432). coords(wk2gt_1,115,53,432).  visible(wk2gt_1,432). orientation(wk2gt_2,141,432). coords(wk2gt_2,98,63,432).  visible(wk2gt_2,432).
orientation(wk2gt_1,125,433). coords(wk2gt_1,115,53,433).  visible(wk2gt_1,433). orientation(wk2gt_2,141,433). coords(wk2gt_2,98,63,433).  visible(wk2gt_2,433).
orientation(wk2gt_1,125,434). coords(wk2gt_1,115,53,434).  visible(wk2gt_1,434). orientation(wk2gt_2,141,434). coords(wk2gt_2,98,63,434).  visible(wk2gt_2,434).
orientation(wk2gt_1,125,435). coords(wk2gt_1,115,53,435).  visible(wk2gt_1,435). orientation(wk2gt_2,141,435). coords(wk2gt_2,98,63,435).  visible(wk2gt_2,435).
orientation(wk2gt_1,125,436). coords(wk2gt_1,115,53,436).  visible(wk2gt_1,436). orientation(wk2gt_2,141,436). coords(wk2gt_2,98,63,436).  visible(wk2gt_2,436).
orientation(wk2gt_1,125,437). coords(wk2gt_1,115,53,437).  visible(wk2gt_1,437). orientation(wk2gt_2,141,437). coords(wk2gt_2,98,63,437).  visible(wk2gt_2,437).
orientation(wk2gt_1,125,438). coords(wk2gt_1,115,53,438).  visible(wk2gt_1,438). orientation(wk2gt_2,141,438). coords(wk2gt_2,98,63,438).  visible(wk2gt_2,438).
orientation(wk2gt_1,125,439). coords(wk2gt_1,115,53,439).  visible(wk2gt_1,439). orientation(wk2gt_2,141,439). coords(wk2gt_2,98,63,439).  visible(wk2gt_2,439).
orientation(wk2gt_1,125,440). coords(wk2gt_1,115,53,440).  visible(wk2gt_1,440). orientation(wk2gt_2,141,440). coords(wk2gt_2,98,63,440).  visible(wk2gt_2,440).
orientation(wk2gt_1,125,441). coords(wk2gt_1,115,53,441).  visible(wk2gt_1,441). orientation(wk2gt_2,141,441). coords(wk2gt_2,98,63,441).  visible(wk2gt_2,441).
orientation(wk2gt_1,125,442). coords(wk2gt_1,115,53,442).  visible(wk2gt_1,442). orientation(wk2gt_2,141,442). coords(wk2gt_2,98,63,442).  visible(wk2gt_2,442).
orientation(wk2gt_1,125,443). coords(wk2gt_1,115,53,443).  visible(wk2gt_1,443). orientation(wk2gt_2,141,443). coords(wk2gt_2,98,63,443).  visible(wk2gt_2,443).
orientation(wk2gt_1,125,444). coords(wk2gt_1,115,53,444).  visible(wk2gt_1,444). orientation(wk2gt_2,141,444). coords(wk2gt_2,98,63,444).  visible(wk2gt_2,444).
orientation(wk2gt_1,125,445). coords(wk2gt_1,115,53,445).  visible(wk2gt_1,445). orientation(wk2gt_2,141,445). coords(wk2gt_2,98,63,445).  visible(wk2gt_2,445).
orientation(wk2gt_1,125,446). coords(wk2gt_1,115,53,446).  visible(wk2gt_1,446). orientation(wk2gt_2,141,446). coords(wk2gt_2,98,63,446).  visible(wk2gt_2,446).
orientation(wk2gt_1,125,447). coords(wk2gt_1,115,53,447).  visible(wk2gt_1,447). orientation(wk2gt_2,141,447). coords(wk2gt_2,98,63,447).  visible(wk2gt_2,447).
orientation(wk2gt_1,125,448). coords(wk2gt_1,115,53,448).  visible(wk2gt_1,448). orientation(wk2gt_2,141,448). coords(wk2gt_2,98,63,448).  visible(wk2gt_2,448).
orientation(wk2gt_1,125,449). coords(wk2gt_1,115,53,449).  visible(wk2gt_1,449). orientation(wk2gt_2,141,449). coords(wk2gt_2,98,63,449).  visible(wk2gt_2,449).
orientation(wk2gt_1,125,450). coords(wk2gt_1,115,53,450).  visible(wk2gt_1,450). orientation(wk2gt_2,141,450). coords(wk2gt_2,98,63,450).  visible(wk2gt_2,450).
orientation(wk2gt_1,125,451). coords(wk2gt_1,115,53,451).  visible(wk2gt_1,451). orientation(wk2gt_2,141,451). coords(wk2gt_2,98,63,451).  visible(wk2gt_2,451).
orientation(wk2gt_1,125,452). coords(wk2gt_1,115,53,452).  visible(wk2gt_1,452). orientation(wk2gt_2,141,452). coords(wk2gt_2,98,63,452).  visible(wk2gt_2,452).
orientation(wk2gt_1,125,453). coords(wk2gt_1,115,53,453).  visible(wk2gt_1,453). orientation(wk2gt_2,141,453). coords(wk2gt_2,98,63,453).  visible(wk2gt_2,453).
orientation(wk2gt_1,125,454). coords(wk2gt_1,115,53,454).  visible(wk2gt_1,454). orientation(wk2gt_2,141,454). coords(wk2gt_2,98,63,454).  visible(wk2gt_2,454).
orientation(wk2gt_1,125,455). coords(wk2gt_1,115,53,455).  visible(wk2gt_1,455). orientation(wk2gt_2,141,455). coords(wk2gt_2,98,63,455).  visible(wk2gt_2,455).
orientation(wk2gt_1,125,456). coords(wk2gt_1,115,53,456).  visible(wk2gt_1,456). orientation(wk2gt_2,141,456). coords(wk2gt_2,98,63,456).  visible(wk2gt_2,456).
orientation(wk2gt_1,125,457). coords(wk2gt_1,115,53,457).  visible(wk2gt_1,457). orientation(wk2gt_2,141,457). coords(wk2gt_2,98,63,457).  visible(wk2gt_2,457).
happensAt(disappear(wk2gt_1),456). happensAt(disappear(wk2gt_2),456).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=432..456.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=432..456.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),432). :- holdsAt(meeting(wk2gt_2,wk2gt_1),432).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),433). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),433).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),434). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),434).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),435). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),435).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),436). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),436).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),437). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),437).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),438). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),438).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),439). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),439).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),440). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),440).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),441). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),441).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),442). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),442).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),443). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),443).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),444). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),444).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),445). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),445).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),446). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),446).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),447). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),447).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),448). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),448).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),449). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),449).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),450). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),450).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),451). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),451).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),452). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),452).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),453). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),453).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),454). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),454).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),455). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),455).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),456). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),456).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),457). :- holdsAt(meeting(wk2gt_2,wk2gt_1),457).
:- simpleEvent(T,P,walking), T>432 , T<456.
:- simpleEvent(T,P,running), T>432 , T<456.

#show simpleEvent/3.