

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

	person(ms3ggt_0;ms3ggt_1).
orientation(ms3ggt_0,158,350). coords(ms3ggt_0,120,145,350).  visible(ms3ggt_0,350). orientation(ms3ggt_1,156,350). coords(ms3ggt_1,114,164,350).  visible(ms3ggt_1,350).
orientation(ms3ggt_0,160,351). coords(ms3ggt_0,121,145,351).  visible(ms3ggt_0,351). orientation(ms3ggt_1,156,351). coords(ms3ggt_1,115,164,351).  visible(ms3ggt_1,351).
orientation(ms3ggt_0,160,352). coords(ms3ggt_0,121,145,352).  visible(ms3ggt_0,352). orientation(ms3ggt_1,156,352). coords(ms3ggt_1,115,163,352).  visible(ms3ggt_1,352).
orientation(ms3ggt_0,160,353). coords(ms3ggt_0,121,145,353).  visible(ms3ggt_0,353). orientation(ms3ggt_1,156,353). coords(ms3ggt_1,115,163,353).  visible(ms3ggt_1,353).
orientation(ms3ggt_0,160,354). coords(ms3ggt_0,121,145,354).  visible(ms3ggt_0,354). orientation(ms3ggt_1,156,354). coords(ms3ggt_1,115,163,354).  visible(ms3ggt_1,354).
orientation(ms3ggt_0,157,355). coords(ms3ggt_0,121,144,355).  visible(ms3ggt_0,355). orientation(ms3ggt_1,156,355). coords(ms3ggt_1,116,162,355).  visible(ms3ggt_1,355).
orientation(ms3ggt_0,157,356). coords(ms3ggt_0,122,144,356).  visible(ms3ggt_0,356). orientation(ms3ggt_1,156,356). coords(ms3ggt_1,116,162,356).  visible(ms3ggt_1,356).
orientation(ms3ggt_0,157,357). coords(ms3ggt_0,123,144,357).  visible(ms3ggt_0,357). orientation(ms3ggt_1,156,357). coords(ms3ggt_1,117,161,357).  visible(ms3ggt_1,357).
orientation(ms3ggt_0,157,358). coords(ms3ggt_0,125,144,358).  visible(ms3ggt_0,358). orientation(ms3ggt_1,156,358). coords(ms3ggt_1,117,161,358).  visible(ms3ggt_1,358).
orientation(ms3ggt_0,157,359). coords(ms3ggt_0,127,144,359).  visible(ms3ggt_0,359). orientation(ms3ggt_1,156,359). coords(ms3ggt_1,117,161,359).  visible(ms3ggt_1,359).
orientation(ms3ggt_0,161,360). coords(ms3ggt_0,129,144,360).  visible(ms3ggt_0,360). orientation(ms3ggt_1,156,360). coords(ms3ggt_1,117,161,360).  visible(ms3ggt_1,360).
orientation(ms3ggt_0,161,361). coords(ms3ggt_0,130,143,361).  visible(ms3ggt_0,361). orientation(ms3ggt_1,156,361). coords(ms3ggt_1,117,159,361).  visible(ms3ggt_1,361).
orientation(ms3ggt_0,161,362). coords(ms3ggt_0,131,143,362).  visible(ms3ggt_0,362). orientation(ms3ggt_1,156,362). coords(ms3ggt_1,118,159,362).  visible(ms3ggt_1,362).
orientation(ms3ggt_0,161,363). coords(ms3ggt_0,133,143,363).  visible(ms3ggt_0,363). orientation(ms3ggt_1,156,363). coords(ms3ggt_1,121,159,363).  visible(ms3ggt_1,363).
orientation(ms3ggt_0,158,364). coords(ms3ggt_0,133,143,364).  visible(ms3ggt_0,364). orientation(ms3ggt_1,156,364). coords(ms3ggt_1,123,159,364).  visible(ms3ggt_1,364).
orientation(ms3ggt_0,158,365). coords(ms3ggt_0,133,142,365).  visible(ms3ggt_0,365). orientation(ms3ggt_1,156,365). coords(ms3ggt_1,124,159,365).  visible(ms3ggt_1,365).
orientation(ms3ggt_0,158,366). coords(ms3ggt_0,134,142,366).  visible(ms3ggt_0,366). orientation(ms3ggt_1,156,366). coords(ms3ggt_1,125,158,366).  visible(ms3ggt_1,366).
orientation(ms3ggt_0,158,367). coords(ms3ggt_0,134,142,367).  visible(ms3ggt_0,367). orientation(ms3ggt_1,156,367). coords(ms3ggt_1,125,158,367).  visible(ms3ggt_1,367).
orientation(ms3ggt_0,158,368). coords(ms3ggt_0,134,142,368).  visible(ms3ggt_0,368). orientation(ms3ggt_1,156,368). coords(ms3ggt_1,126,158,368).  visible(ms3ggt_1,368).
orientation(ms3ggt_0,158,369). coords(ms3ggt_0,134,142,369).  visible(ms3ggt_0,369). orientation(ms3ggt_1,156,369). coords(ms3ggt_1,127,158,369).  visible(ms3ggt_1,369).
orientation(ms3ggt_0,158,370). coords(ms3ggt_0,135,142,370).  visible(ms3ggt_0,370). orientation(ms3ggt_1,156,370). coords(ms3ggt_1,127,158,370).  visible(ms3ggt_1,370).
orientation(ms3ggt_0,158,371). coords(ms3ggt_0,136,142,371).  visible(ms3ggt_0,371). orientation(ms3ggt_1,156,371). coords(ms3ggt_1,128,158,371).  visible(ms3ggt_1,371).
orientation(ms3ggt_0,158,372). coords(ms3ggt_0,136,142,372).  visible(ms3ggt_0,372). orientation(ms3ggt_1,156,372). coords(ms3ggt_1,128,158,372).  visible(ms3ggt_1,372).
orientation(ms3ggt_0,158,373). coords(ms3ggt_0,137,142,373).  visible(ms3ggt_0,373). orientation(ms3ggt_1,156,373). coords(ms3ggt_1,129,158,373).  visible(ms3ggt_1,373).
orientation(ms3ggt_0,158,374). coords(ms3ggt_0,137,142,374).  visible(ms3ggt_0,374). orientation(ms3ggt_1,156,374). coords(ms3ggt_1,130,158,374).  visible(ms3ggt_1,374).
orientation(ms3ggt_0,158,375). coords(ms3ggt_0,137,142,375).  visible(ms3ggt_0,375). orientation(ms3ggt_1,156,375). coords(ms3ggt_1,130,158,375).  visible(ms3ggt_1,375).
happensAt(disappear(ms3ggt_0),374). happensAt(disappear(ms3ggt_1),374).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=350..374.
1{simpleEvent(T,ms3ggt_1,active); simpleEvent(T,ms3ggt_1,inactive); simpleEvent(T,ms3ggt_1,walking); simpleEvent(T,ms3ggt_1,running)}1 :- T=350..374.
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),350). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),350).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),351). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),351).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),352). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),352).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),353). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),353).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),354). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),354).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),355). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),355).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),356). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),356).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),357). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),357).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),358). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),358).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),359). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),359).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),360). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),360).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),361). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),361).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),362). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),362).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),363). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),363).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),364). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),364).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),365). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),365).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),366). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),366).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),367). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),367).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),368). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),368).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),369). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),369).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),370). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),370).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),371). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),371).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),372). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),372).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),373). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),373).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),374). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),374).
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),375). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),375).
:- simpleEvent(T,P,inactive), T>350 , T<374.
:- simpleEvent(T,P,running), T>350 , T<374.

#show simpleEvent/3.