

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
orientation(ms3ggt_0,158,376). coords(ms3ggt_0,137,142,376).  visible(ms3ggt_0,376). orientation(ms3ggt_1,156,376). coords(ms3ggt_1,130,158,376).  visible(ms3ggt_1,376).
orientation(ms3ggt_0,158,377). coords(ms3ggt_0,139,142,377).  visible(ms3ggt_0,377). orientation(ms3ggt_1,156,377). coords(ms3ggt_1,132,158,377).  visible(ms3ggt_1,377).
orientation(ms3ggt_0,158,378). coords(ms3ggt_0,140,142,378).  visible(ms3ggt_0,378). orientation(ms3ggt_1,156,378). coords(ms3ggt_1,134,158,378).  visible(ms3ggt_1,378).
orientation(ms3ggt_0,158,379). coords(ms3ggt_0,140,142,379).  visible(ms3ggt_0,379). orientation(ms3ggt_1,162,379). coords(ms3ggt_1,136,158,379).  visible(ms3ggt_1,379).
orientation(ms3ggt_0,158,380). coords(ms3ggt_0,141,142,380).  visible(ms3ggt_0,380). orientation(ms3ggt_1,162,380). coords(ms3ggt_1,138,158,380).  visible(ms3ggt_1,380).
orientation(ms3ggt_0,152,381). coords(ms3ggt_0,141,142,381).  visible(ms3ggt_0,381). orientation(ms3ggt_1,162,381). coords(ms3ggt_1,138,158,381).  visible(ms3ggt_1,381).
orientation(ms3ggt_0,152,382). coords(ms3ggt_0,142,141,382).  visible(ms3ggt_0,382). orientation(ms3ggt_1,162,382). coords(ms3ggt_1,140,158,382).  visible(ms3ggt_1,382).
orientation(ms3ggt_0,152,383). coords(ms3ggt_0,142,141,383).  visible(ms3ggt_0,383). orientation(ms3ggt_1,162,383). coords(ms3ggt_1,141,158,383).  visible(ms3ggt_1,383).
orientation(ms3ggt_0,152,384). coords(ms3ggt_0,142,140,384).  visible(ms3ggt_0,384). orientation(ms3ggt_1,162,384). coords(ms3ggt_1,142,158,384).  visible(ms3ggt_1,384).
orientation(ms3ggt_0,152,385). coords(ms3ggt_0,142,140,385).  visible(ms3ggt_0,385). orientation(ms3ggt_1,162,385). coords(ms3ggt_1,142,157,385).  visible(ms3ggt_1,385).
orientation(ms3ggt_0,152,386). coords(ms3ggt_0,142,140,386).  visible(ms3ggt_0,386). orientation(ms3ggt_1,162,386). coords(ms3ggt_1,142,157,386).  visible(ms3ggt_1,386).
orientation(ms3ggt_0,152,387). coords(ms3ggt_0,142,140,387).  visible(ms3ggt_0,387). orientation(ms3ggt_1,162,387). coords(ms3ggt_1,142,157,387).  visible(ms3ggt_1,387).
happensAt(disappear(ms3ggt_0),386). happensAt(disappear(ms3ggt_1),386).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=362..386.
1{simpleEvent(T,ms3ggt_1,active); simpleEvent(T,ms3ggt_1,inactive); simpleEvent(T,ms3ggt_1,walking); simpleEvent(T,ms3ggt_1,running)}1 :- T=362..386.
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),362). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),362).
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
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),375). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),375).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),376). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),376).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),377). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),377).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),378). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),378).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),379). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),379).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),380). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),380).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),381). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),381).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),382). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),382).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),383). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),383).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),384). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),384).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),385). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),385).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_1),386). :- not holdsAt(moving(ms3ggt_1,ms3ggt_0),386).
:- holdsAt(moving(ms3ggt_0,ms3ggt_1),387). :- holdsAt(moving(ms3ggt_1,ms3ggt_0),387).
:- simpleEvent(T,P,inactive), T>362 , T<386.
:- simpleEvent(T,P,running), T>362 , T<386.

#show simpleEvent/3.