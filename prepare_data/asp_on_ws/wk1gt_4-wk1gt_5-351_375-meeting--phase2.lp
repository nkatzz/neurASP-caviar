

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,167,351). coords(wk1gt_4,52,176,351).  visible(wk1gt_4,351). orientation(wk1gt_5,162,351). coords(wk1gt_5,50,163,351).  visible(wk1gt_5,351).
orientation(wk1gt_4,162,352). coords(wk1gt_4,52,176,352).  visible(wk1gt_4,352). orientation(wk1gt_5,162,352). coords(wk1gt_5,50,163,352).  visible(wk1gt_5,352).
orientation(wk1gt_4,159,353). coords(wk1gt_4,52,176,353).  visible(wk1gt_4,353). orientation(wk1gt_5,162,353). coords(wk1gt_5,50,163,353).  visible(wk1gt_5,353).
orientation(wk1gt_4,162,354). coords(wk1gt_4,52,176,354).  visible(wk1gt_4,354). orientation(wk1gt_5,162,354). coords(wk1gt_5,50,163,354).  visible(wk1gt_5,354).
orientation(wk1gt_4,165,355). coords(wk1gt_4,52,176,355).  visible(wk1gt_4,355). orientation(wk1gt_5,162,355). coords(wk1gt_5,50,163,355).  visible(wk1gt_5,355).
orientation(wk1gt_4,158,356). coords(wk1gt_4,52,176,356).  visible(wk1gt_4,356). orientation(wk1gt_5,162,356). coords(wk1gt_5,50,163,356).  visible(wk1gt_5,356).
orientation(wk1gt_4,157,357). coords(wk1gt_4,52,176,357).  visible(wk1gt_4,357). orientation(wk1gt_5,162,357). coords(wk1gt_5,50,163,357).  visible(wk1gt_5,357).
orientation(wk1gt_4,161,358). coords(wk1gt_4,52,176,358).  visible(wk1gt_4,358). orientation(wk1gt_5,162,358). coords(wk1gt_5,50,163,358).  visible(wk1gt_5,358).
orientation(wk1gt_4,161,359). coords(wk1gt_4,52,176,359).  visible(wk1gt_4,359). orientation(wk1gt_5,162,359). coords(wk1gt_5,50,163,359).  visible(wk1gt_5,359).
orientation(wk1gt_4,162,360). coords(wk1gt_4,52,176,360).  visible(wk1gt_4,360). orientation(wk1gt_5,162,360). coords(wk1gt_5,50,163,360).  visible(wk1gt_5,360).
orientation(wk1gt_4,163,361). coords(wk1gt_4,52,176,361).  visible(wk1gt_4,361). orientation(wk1gt_5,162,361). coords(wk1gt_5,50,163,361).  visible(wk1gt_5,361).
orientation(wk1gt_4,168,362). coords(wk1gt_4,52,176,362).  visible(wk1gt_4,362). orientation(wk1gt_5,162,362). coords(wk1gt_5,50,163,362).  visible(wk1gt_5,362).
orientation(wk1gt_4,164,363). coords(wk1gt_4,52,176,363).  visible(wk1gt_4,363). orientation(wk1gt_5,162,363). coords(wk1gt_5,50,163,363).  visible(wk1gt_5,363).
orientation(wk1gt_4,166,364). coords(wk1gt_4,52,176,364).  visible(wk1gt_4,364). orientation(wk1gt_5,162,364). coords(wk1gt_5,51,163,364).  visible(wk1gt_5,364).
orientation(wk1gt_4,162,365). coords(wk1gt_4,52,176,365).  visible(wk1gt_4,365). orientation(wk1gt_5,162,365). coords(wk1gt_5,50,163,365).  visible(wk1gt_5,365).
orientation(wk1gt_4,164,366). coords(wk1gt_4,52,176,366).  visible(wk1gt_4,366). orientation(wk1gt_5,162,366). coords(wk1gt_5,50,163,366).  visible(wk1gt_5,366).
orientation(wk1gt_4,162,367). coords(wk1gt_4,52,176,367).  visible(wk1gt_4,367). orientation(wk1gt_5,162,367). coords(wk1gt_5,50,163,367).  visible(wk1gt_5,367).
orientation(wk1gt_4,161,368). coords(wk1gt_4,52,176,368).  visible(wk1gt_4,368). orientation(wk1gt_5,162,368). coords(wk1gt_5,51,163,368).  visible(wk1gt_5,368).
orientation(wk1gt_4,162,369). coords(wk1gt_4,52,176,369).  visible(wk1gt_4,369). orientation(wk1gt_5,162,369). coords(wk1gt_5,50,163,369).  visible(wk1gt_5,369).
orientation(wk1gt_4,163,370). coords(wk1gt_4,52,176,370).  visible(wk1gt_4,370). orientation(wk1gt_5,162,370). coords(wk1gt_5,50,163,370).  visible(wk1gt_5,370).
orientation(wk1gt_4,164,371). coords(wk1gt_4,52,176,371).  visible(wk1gt_4,371). orientation(wk1gt_5,162,371). coords(wk1gt_5,50,163,371).  visible(wk1gt_5,371).
orientation(wk1gt_4,163,372). coords(wk1gt_4,52,176,372).  visible(wk1gt_4,372). orientation(wk1gt_5,162,372). coords(wk1gt_5,50,163,372).  visible(wk1gt_5,372).
orientation(wk1gt_4,164,373). coords(wk1gt_4,52,176,373).  visible(wk1gt_4,373). orientation(wk1gt_5,162,373). coords(wk1gt_5,50,163,373).  visible(wk1gt_5,373).
orientation(wk1gt_4,166,374). coords(wk1gt_4,52,176,374).  visible(wk1gt_4,374). orientation(wk1gt_5,162,374). coords(wk1gt_5,50,163,374).  visible(wk1gt_5,374).
orientation(wk1gt_4,170,375). coords(wk1gt_4,52,176,375).  visible(wk1gt_4,375). orientation(wk1gt_5,162,375). coords(wk1gt_5,50,163,375).  visible(wk1gt_5,375).
orientation(wk1gt_4,170,376). coords(wk1gt_4,52,176,376).  visible(wk1gt_4,376). orientation(wk1gt_5,162,376). coords(wk1gt_5,50,163,376).  visible(wk1gt_5,376).
happensAt(disappear(wk1gt_4),375). happensAt(disappear(wk1gt_5),375).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=351..375.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=351..375.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),351). :- holdsAt(meeting(wk1gt_5,wk1gt_4),351).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),352). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),352).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),353). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),353).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),354). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),354).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),355). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),355).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),356). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),356).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),357). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),357).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),358). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),358).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),359). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),359).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),360). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),360).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),361). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),361).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),362). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),362).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),363). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),363).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),364). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),364).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),365). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),365).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),366). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),366).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),367). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),367).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),368). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),368).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),369). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),369).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),370). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),370).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),371). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),371).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),372). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),372).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),373). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),373).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),374). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),374).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),375). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),375).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),376). :- holdsAt(meeting(wk1gt_5,wk1gt_4),376).
:- simpleEvent(T,P,walking), T>351 , T<375.
:- simpleEvent(T,P,running), T>351 , T<375.

#show simpleEvent/3.