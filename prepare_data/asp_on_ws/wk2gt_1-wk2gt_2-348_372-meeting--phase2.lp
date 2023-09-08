

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
orientation(wk2gt_1,125,348). coords(wk2gt_1,115,53,348).  visible(wk2gt_1,348). orientation(wk2gt_2,141,348). coords(wk2gt_2,98,63,348).  visible(wk2gt_2,348).
orientation(wk2gt_1,125,349). coords(wk2gt_1,115,53,349).  visible(wk2gt_1,349). orientation(wk2gt_2,141,349). coords(wk2gt_2,98,63,349).  visible(wk2gt_2,349).
orientation(wk2gt_1,125,350). coords(wk2gt_1,115,53,350).  visible(wk2gt_1,350). orientation(wk2gt_2,141,350). coords(wk2gt_2,98,63,350).  visible(wk2gt_2,350).
orientation(wk2gt_1,125,351). coords(wk2gt_1,115,53,351).  visible(wk2gt_1,351). orientation(wk2gt_2,141,351). coords(wk2gt_2,98,63,351).  visible(wk2gt_2,351).
orientation(wk2gt_1,125,352). coords(wk2gt_1,115,53,352).  visible(wk2gt_1,352). orientation(wk2gt_2,141,352). coords(wk2gt_2,98,63,352).  visible(wk2gt_2,352).
orientation(wk2gt_1,125,353). coords(wk2gt_1,115,53,353).  visible(wk2gt_1,353). orientation(wk2gt_2,141,353). coords(wk2gt_2,98,63,353).  visible(wk2gt_2,353).
orientation(wk2gt_1,125,354). coords(wk2gt_1,115,53,354).  visible(wk2gt_1,354). orientation(wk2gt_2,141,354). coords(wk2gt_2,98,63,354).  visible(wk2gt_2,354).
orientation(wk2gt_1,125,355). coords(wk2gt_1,115,53,355).  visible(wk2gt_1,355). orientation(wk2gt_2,141,355). coords(wk2gt_2,98,63,355).  visible(wk2gt_2,355).
orientation(wk2gt_1,125,356). coords(wk2gt_1,115,53,356).  visible(wk2gt_1,356). orientation(wk2gt_2,141,356). coords(wk2gt_2,98,63,356).  visible(wk2gt_2,356).
orientation(wk2gt_1,125,357). coords(wk2gt_1,115,53,357).  visible(wk2gt_1,357). orientation(wk2gt_2,141,357). coords(wk2gt_2,98,63,357).  visible(wk2gt_2,357).
orientation(wk2gt_1,125,358). coords(wk2gt_1,115,53,358).  visible(wk2gt_1,358). orientation(wk2gt_2,141,358). coords(wk2gt_2,98,63,358).  visible(wk2gt_2,358).
orientation(wk2gt_1,125,359). coords(wk2gt_1,115,53,359).  visible(wk2gt_1,359). orientation(wk2gt_2,141,359). coords(wk2gt_2,98,63,359).  visible(wk2gt_2,359).
orientation(wk2gt_1,125,360). coords(wk2gt_1,115,53,360).  visible(wk2gt_1,360). orientation(wk2gt_2,141,360). coords(wk2gt_2,98,63,360).  visible(wk2gt_2,360).
orientation(wk2gt_1,125,361). coords(wk2gt_1,115,53,361).  visible(wk2gt_1,361). orientation(wk2gt_2,141,361). coords(wk2gt_2,98,63,361).  visible(wk2gt_2,361).
orientation(wk2gt_1,125,362). coords(wk2gt_1,115,53,362).  visible(wk2gt_1,362). orientation(wk2gt_2,141,362). coords(wk2gt_2,98,63,362).  visible(wk2gt_2,362).
orientation(wk2gt_1,125,363). coords(wk2gt_1,115,53,363).  visible(wk2gt_1,363). orientation(wk2gt_2,141,363). coords(wk2gt_2,98,63,363).  visible(wk2gt_2,363).
orientation(wk2gt_1,125,364). coords(wk2gt_1,115,53,364).  visible(wk2gt_1,364). orientation(wk2gt_2,141,364). coords(wk2gt_2,98,63,364).  visible(wk2gt_2,364).
orientation(wk2gt_1,125,365). coords(wk2gt_1,115,53,365).  visible(wk2gt_1,365). orientation(wk2gt_2,141,365). coords(wk2gt_2,98,63,365).  visible(wk2gt_2,365).
orientation(wk2gt_1,125,366). coords(wk2gt_1,115,53,366).  visible(wk2gt_1,366). orientation(wk2gt_2,141,366). coords(wk2gt_2,98,63,366).  visible(wk2gt_2,366).
orientation(wk2gt_1,125,367). coords(wk2gt_1,115,53,367).  visible(wk2gt_1,367). orientation(wk2gt_2,141,367). coords(wk2gt_2,98,63,367).  visible(wk2gt_2,367).
orientation(wk2gt_1,125,368). coords(wk2gt_1,115,53,368).  visible(wk2gt_1,368). orientation(wk2gt_2,141,368). coords(wk2gt_2,98,63,368).  visible(wk2gt_2,368).
orientation(wk2gt_1,125,369). coords(wk2gt_1,115,53,369).  visible(wk2gt_1,369). orientation(wk2gt_2,141,369). coords(wk2gt_2,98,63,369).  visible(wk2gt_2,369).
orientation(wk2gt_1,125,370). coords(wk2gt_1,115,53,370).  visible(wk2gt_1,370). orientation(wk2gt_2,141,370). coords(wk2gt_2,98,63,370).  visible(wk2gt_2,370).
orientation(wk2gt_1,125,371). coords(wk2gt_1,115,53,371).  visible(wk2gt_1,371). orientation(wk2gt_2,141,371). coords(wk2gt_2,98,63,371).  visible(wk2gt_2,371).
orientation(wk2gt_1,125,372). coords(wk2gt_1,115,53,372).  visible(wk2gt_1,372). orientation(wk2gt_2,141,372). coords(wk2gt_2,98,63,372).  visible(wk2gt_2,372).
orientation(wk2gt_1,125,373). coords(wk2gt_1,115,53,373).  visible(wk2gt_1,373). orientation(wk2gt_2,141,373). coords(wk2gt_2,98,63,373).  visible(wk2gt_2,373).
happensAt(disappear(wk2gt_1),372). happensAt(disappear(wk2gt_2),372).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=348..372.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=348..372.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),348). :- holdsAt(meeting(wk2gt_2,wk2gt_1),348).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),349). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),349).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),350). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),350).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),351). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),351).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),352). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),352).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),353). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),353).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),354). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),354).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),355). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),355).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),356). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),356).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),357). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),357).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),358). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),358).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),359). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),359).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),360). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),360).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),361). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),361).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),362). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),362).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),363). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),363).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),364). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),364).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),365). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),365).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),366). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),366).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),367). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),367).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),368). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),368).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),369). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),369).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),370). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),370).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),371). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),371).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),372). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),372).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),373). :- holdsAt(meeting(wk2gt_2,wk2gt_1),373).
:- simpleEvent(T,P,walking), T>348 , T<372.
:- simpleEvent(T,P,running), T>348 , T<372.

#show simpleEvent/3.