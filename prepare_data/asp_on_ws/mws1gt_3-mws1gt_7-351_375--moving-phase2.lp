

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

	person(mws1gt_3;mws1gt_7).
orientation(mws1gt_3,165,351). coords(mws1gt_3,53,154,351).  visible(mws1gt_3,351). orientation(mws1gt_7,174,351). coords(mws1gt_7,46,188,351).  visible(mws1gt_7,351).
orientation(mws1gt_3,165,352). coords(mws1gt_3,53,154,352).  visible(mws1gt_3,352). orientation(mws1gt_7,174,352). coords(mws1gt_7,46,188,352).  visible(mws1gt_7,352).
orientation(mws1gt_3,165,353). coords(mws1gt_3,53,154,353).  visible(mws1gt_3,353). orientation(mws1gt_7,174,353). coords(mws1gt_7,46,188,353).  visible(mws1gt_7,353).
orientation(mws1gt_3,165,354). coords(mws1gt_3,52,154,354).  visible(mws1gt_3,354). orientation(mws1gt_7,174,354). coords(mws1gt_7,46,188,354).  visible(mws1gt_7,354).
orientation(mws1gt_3,165,355). coords(mws1gt_3,51,154,355).  visible(mws1gt_3,355). orientation(mws1gt_7,174,355). coords(mws1gt_7,46,187,355).  visible(mws1gt_7,355).
orientation(mws1gt_3,165,356). coords(mws1gt_3,51,155,356).  visible(mws1gt_3,356). orientation(mws1gt_7,174,356). coords(mws1gt_7,46,187,356).  visible(mws1gt_7,356).
orientation(mws1gt_3,165,357). coords(mws1gt_3,50,155,357).  visible(mws1gt_3,357). orientation(mws1gt_7,174,357). coords(mws1gt_7,47,187,357).  visible(mws1gt_7,357).
orientation(mws1gt_3,165,358). coords(mws1gt_3,50,157,358).  visible(mws1gt_3,358). orientation(mws1gt_7,174,358). coords(mws1gt_7,47,187,358).  visible(mws1gt_7,358).
orientation(mws1gt_3,165,359). coords(mws1gt_3,49,157,359).  visible(mws1gt_3,359). orientation(mws1gt_7,174,359). coords(mws1gt_7,49,186,359).  visible(mws1gt_7,359).
orientation(mws1gt_3,165,360). coords(mws1gt_3,49,158,360).  visible(mws1gt_3,360). orientation(mws1gt_7,174,360). coords(mws1gt_7,50,185,360).  visible(mws1gt_7,360).
orientation(mws1gt_3,165,361). coords(mws1gt_3,49,159,361).  visible(mws1gt_3,361). orientation(mws1gt_7,174,361). coords(mws1gt_7,50,185,361).  visible(mws1gt_7,361).
orientation(mws1gt_3,165,362). coords(mws1gt_3,49,159,362).  visible(mws1gt_3,362). orientation(mws1gt_7,174,362). coords(mws1gt_7,50,185,362).  visible(mws1gt_7,362).
orientation(mws1gt_3,165,363). coords(mws1gt_3,49,159,363).  visible(mws1gt_3,363). orientation(mws1gt_7,174,363). coords(mws1gt_7,50,185,363).  visible(mws1gt_7,363).
orientation(mws1gt_3,165,364). coords(mws1gt_3,48,159,364).  visible(mws1gt_3,364). orientation(mws1gt_7,174,364). coords(mws1gt_7,50,185,364).  visible(mws1gt_7,364).
orientation(mws1gt_3,165,365). coords(mws1gt_3,47,159,365).  visible(mws1gt_3,365). orientation(mws1gt_7,174,365). coords(mws1gt_7,50,185,365).  visible(mws1gt_7,365).
orientation(mws1gt_3,165,366). coords(mws1gt_3,47,159,366).  visible(mws1gt_3,366). orientation(mws1gt_7,174,366). coords(mws1gt_7,50,185,366).  visible(mws1gt_7,366).
orientation(mws1gt_3,165,367). coords(mws1gt_3,47,159,367).  visible(mws1gt_3,367). orientation(mws1gt_7,174,367). coords(mws1gt_7,50,185,367).  visible(mws1gt_7,367).
orientation(mws1gt_3,165,368). coords(mws1gt_3,46,159,368).  visible(mws1gt_3,368). orientation(mws1gt_7,174,368). coords(mws1gt_7,50,185,368).  visible(mws1gt_7,368).
orientation(mws1gt_3,165,369). coords(mws1gt_3,45,160,369).  visible(mws1gt_3,369). orientation(mws1gt_7,174,369). coords(mws1gt_7,50,185,369).  visible(mws1gt_7,369).
orientation(mws1gt_3,165,370). coords(mws1gt_3,45,160,370).  visible(mws1gt_3,370). orientation(mws1gt_7,174,370). coords(mws1gt_7,50,185,370).  visible(mws1gt_7,370).
orientation(mws1gt_3,165,371). coords(mws1gt_3,44,160,371).  visible(mws1gt_3,371). orientation(mws1gt_7,174,371). coords(mws1gt_7,51,185,371).  visible(mws1gt_7,371).
orientation(mws1gt_3,165,372). coords(mws1gt_3,44,160,372).  visible(mws1gt_3,372). orientation(mws1gt_7,174,372). coords(mws1gt_7,53,185,372).  visible(mws1gt_7,372).
orientation(mws1gt_3,165,373). coords(mws1gt_3,44,160,373).  visible(mws1gt_3,373). orientation(mws1gt_7,174,373). coords(mws1gt_7,53,185,373).  visible(mws1gt_7,373).
orientation(mws1gt_3,165,374). coords(mws1gt_3,43,160,374).  visible(mws1gt_3,374). orientation(mws1gt_7,174,374). coords(mws1gt_7,55,185,374).  visible(mws1gt_7,374).
orientation(mws1gt_3,165,375). coords(mws1gt_3,42,160,375).  visible(mws1gt_3,375). orientation(mws1gt_7,174,375). coords(mws1gt_7,56,184,375).  visible(mws1gt_7,375).
orientation(mws1gt_3,165,376). coords(mws1gt_3,42,160,376).  visible(mws1gt_3,376). orientation(mws1gt_7,174,376). coords(mws1gt_7,56,184,376).  visible(mws1gt_7,376).
happensAt(disappear(mws1gt_3),375). happensAt(disappear(mws1gt_7),375).
1{simpleEvent(T,mws1gt_3,active); simpleEvent(T,mws1gt_3,inactive); simpleEvent(T,mws1gt_3,walking); simpleEvent(T,mws1gt_3,running)}1 :- T=351..375.
1{simpleEvent(T,mws1gt_7,active); simpleEvent(T,mws1gt_7,inactive); simpleEvent(T,mws1gt_7,walking); simpleEvent(T,mws1gt_7,running)}1 :- T=351..375.
:- holdsAt(moving(mws1gt_3,mws1gt_7),351). :- holdsAt(moving(mws1gt_7,mws1gt_3),351).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),352). :- not holdsAt(moving(mws1gt_7,mws1gt_3),352).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),353). :- not holdsAt(moving(mws1gt_7,mws1gt_3),353).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),354). :- not holdsAt(moving(mws1gt_7,mws1gt_3),354).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),355). :- not holdsAt(moving(mws1gt_7,mws1gt_3),355).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),356). :- not holdsAt(moving(mws1gt_7,mws1gt_3),356).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),357). :- not holdsAt(moving(mws1gt_7,mws1gt_3),357).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),358). :- not holdsAt(moving(mws1gt_7,mws1gt_3),358).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),359). :- not holdsAt(moving(mws1gt_7,mws1gt_3),359).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),360). :- not holdsAt(moving(mws1gt_7,mws1gt_3),360).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),361). :- not holdsAt(moving(mws1gt_7,mws1gt_3),361).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),362). :- not holdsAt(moving(mws1gt_7,mws1gt_3),362).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),363). :- not holdsAt(moving(mws1gt_7,mws1gt_3),363).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),364). :- not holdsAt(moving(mws1gt_7,mws1gt_3),364).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),365). :- not holdsAt(moving(mws1gt_7,mws1gt_3),365).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),366). :- not holdsAt(moving(mws1gt_7,mws1gt_3),366).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),367). :- not holdsAt(moving(mws1gt_7,mws1gt_3),367).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),368). :- not holdsAt(moving(mws1gt_7,mws1gt_3),368).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),369). :- not holdsAt(moving(mws1gt_7,mws1gt_3),369).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),370). :- not holdsAt(moving(mws1gt_7,mws1gt_3),370).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),371). :- not holdsAt(moving(mws1gt_7,mws1gt_3),371).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),372). :- not holdsAt(moving(mws1gt_7,mws1gt_3),372).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),373). :- not holdsAt(moving(mws1gt_7,mws1gt_3),373).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),374). :- not holdsAt(moving(mws1gt_7,mws1gt_3),374).
:- not holdsAt(moving(mws1gt_3,mws1gt_7),375). :- not holdsAt(moving(mws1gt_7,mws1gt_3),375).
:- holdsAt(moving(mws1gt_3,mws1gt_7),376). :- holdsAt(moving(mws1gt_7,mws1gt_3),376).
:- simpleEvent(T,P,inactive), T>351 , T<375.
:- simpleEvent(T,P,running), T>351 , T<375.

#show simpleEvent/3.