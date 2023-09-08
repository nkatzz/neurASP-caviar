

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

	person(fcgt_3;fcgt_4).
orientation(fcgt_3,164,360). coords(fcgt_3,99,187,360).  visible(fcgt_3,360). orientation(fcgt_4,166,360). coords(fcgt_4,82,206,360).  visible(fcgt_4,360).
orientation(fcgt_3,164,361). coords(fcgt_3,99,187,361).  visible(fcgt_3,361). orientation(fcgt_4,166,361). coords(fcgt_4,82,206,361).  visible(fcgt_4,361).
orientation(fcgt_3,164,362). coords(fcgt_3,99,187,362).  visible(fcgt_3,362). orientation(fcgt_4,166,362). coords(fcgt_4,82,206,362).  visible(fcgt_4,362).
orientation(fcgt_3,164,363). coords(fcgt_3,99,187,363).  visible(fcgt_3,363). orientation(fcgt_4,166,363). coords(fcgt_4,82,206,363).  visible(fcgt_4,363).
orientation(fcgt_3,164,364). coords(fcgt_3,99,187,364).  visible(fcgt_3,364). orientation(fcgt_4,166,364). coords(fcgt_4,82,206,364).  visible(fcgt_4,364).
orientation(fcgt_3,164,365). coords(fcgt_3,99,187,365).  visible(fcgt_3,365). orientation(fcgt_4,166,365). coords(fcgt_4,82,206,365).  visible(fcgt_4,365).
orientation(fcgt_3,164,366). coords(fcgt_3,99,187,366).  visible(fcgt_3,366). orientation(fcgt_4,166,366). coords(fcgt_4,82,206,366).  visible(fcgt_4,366).
orientation(fcgt_3,164,367). coords(fcgt_3,99,187,367).  visible(fcgt_3,367). orientation(fcgt_4,166,367). coords(fcgt_4,82,206,367).  visible(fcgt_4,367).
orientation(fcgt_3,164,368). coords(fcgt_3,99,187,368).  visible(fcgt_3,368). orientation(fcgt_4,166,368). coords(fcgt_4,82,206,368).  visible(fcgt_4,368).
orientation(fcgt_3,164,369). coords(fcgt_3,99,187,369).  visible(fcgt_3,369). orientation(fcgt_4,166,369). coords(fcgt_4,82,206,369).  visible(fcgt_4,369).
orientation(fcgt_3,164,370). coords(fcgt_3,99,187,370).  visible(fcgt_3,370). orientation(fcgt_4,166,370). coords(fcgt_4,82,206,370).  visible(fcgt_4,370).
orientation(fcgt_3,164,371). coords(fcgt_3,99,187,371).  visible(fcgt_3,371). orientation(fcgt_4,166,371). coords(fcgt_4,82,206,371).  visible(fcgt_4,371).
orientation(fcgt_3,164,372). coords(fcgt_3,99,187,372).  visible(fcgt_3,372). orientation(fcgt_4,166,372). coords(fcgt_4,82,206,372).  visible(fcgt_4,372).
orientation(fcgt_3,164,373). coords(fcgt_3,99,187,373).  visible(fcgt_3,373). orientation(fcgt_4,166,373). coords(fcgt_4,82,206,373).  visible(fcgt_4,373).
orientation(fcgt_3,164,374). coords(fcgt_3,99,187,374).  visible(fcgt_3,374). orientation(fcgt_4,166,374). coords(fcgt_4,82,206,374).  visible(fcgt_4,374).
orientation(fcgt_3,164,375). coords(fcgt_3,99,187,375).  visible(fcgt_3,375). orientation(fcgt_4,166,375). coords(fcgt_4,82,206,375).  visible(fcgt_4,375).
orientation(fcgt_3,164,376). coords(fcgt_3,99,187,376).  visible(fcgt_3,376). orientation(fcgt_4,166,376). coords(fcgt_4,82,206,376).  visible(fcgt_4,376).
orientation(fcgt_3,164,377). coords(fcgt_3,99,187,377).  visible(fcgt_3,377). orientation(fcgt_4,166,377). coords(fcgt_4,82,206,377).  visible(fcgt_4,377).
orientation(fcgt_3,164,378). coords(fcgt_3,99,187,378).  visible(fcgt_3,378). orientation(fcgt_4,166,378). coords(fcgt_4,82,206,378).  visible(fcgt_4,378).
orientation(fcgt_3,164,379). coords(fcgt_3,99,187,379).  visible(fcgt_3,379). orientation(fcgt_4,166,379). coords(fcgt_4,82,206,379).  visible(fcgt_4,379).
orientation(fcgt_3,164,380). coords(fcgt_3,99,187,380).  visible(fcgt_3,380). orientation(fcgt_4,166,380). coords(fcgt_4,82,206,380).  visible(fcgt_4,380).
orientation(fcgt_3,164,381). coords(fcgt_3,99,187,381).  visible(fcgt_3,381). orientation(fcgt_4,166,381). coords(fcgt_4,82,206,381).  visible(fcgt_4,381).
orientation(fcgt_3,164,382). coords(fcgt_3,99,187,382).  visible(fcgt_3,382). orientation(fcgt_4,166,382). coords(fcgt_4,82,206,382).  visible(fcgt_4,382).
orientation(fcgt_3,164,383). coords(fcgt_3,99,187,383).  visible(fcgt_3,383). orientation(fcgt_4,166,383). coords(fcgt_4,82,206,383).  visible(fcgt_4,383).
orientation(fcgt_3,164,384). coords(fcgt_3,99,187,384).  visible(fcgt_3,384). orientation(fcgt_4,166,384). coords(fcgt_4,82,206,384).  visible(fcgt_4,384).
orientation(fcgt_3,164,385). coords(fcgt_3,99,187,385).  visible(fcgt_3,385). orientation(fcgt_4,166,385). coords(fcgt_4,82,206,385).  visible(fcgt_4,385).
happensAt(disappear(fcgt_3),384). happensAt(disappear(fcgt_4),384).
1{simpleEvent(T,fcgt_3,active); simpleEvent(T,fcgt_3,inactive); simpleEvent(T,fcgt_3,walking); simpleEvent(T,fcgt_3,running)}1 :- T=360..384.
1{simpleEvent(T,fcgt_4,active); simpleEvent(T,fcgt_4,inactive); simpleEvent(T,fcgt_4,walking); simpleEvent(T,fcgt_4,running)}1 :- T=360..384.
:- holdsAt(meeting(fcgt_3,fcgt_4),360). :- holdsAt(meeting(fcgt_4,fcgt_3),360).
:- not holdsAt(meeting(fcgt_3,fcgt_4),361). :- not holdsAt(meeting(fcgt_4,fcgt_3),361).
:- not holdsAt(meeting(fcgt_3,fcgt_4),362). :- not holdsAt(meeting(fcgt_4,fcgt_3),362).
:- not holdsAt(meeting(fcgt_3,fcgt_4),363). :- not holdsAt(meeting(fcgt_4,fcgt_3),363).
:- not holdsAt(meeting(fcgt_3,fcgt_4),364). :- not holdsAt(meeting(fcgt_4,fcgt_3),364).
:- not holdsAt(meeting(fcgt_3,fcgt_4),365). :- not holdsAt(meeting(fcgt_4,fcgt_3),365).
:- not holdsAt(meeting(fcgt_3,fcgt_4),366). :- not holdsAt(meeting(fcgt_4,fcgt_3),366).
:- not holdsAt(meeting(fcgt_3,fcgt_4),367). :- not holdsAt(meeting(fcgt_4,fcgt_3),367).
:- not holdsAt(meeting(fcgt_3,fcgt_4),368). :- not holdsAt(meeting(fcgt_4,fcgt_3),368).
:- not holdsAt(meeting(fcgt_3,fcgt_4),369). :- not holdsAt(meeting(fcgt_4,fcgt_3),369).
:- not holdsAt(meeting(fcgt_3,fcgt_4),370). :- not holdsAt(meeting(fcgt_4,fcgt_3),370).
:- not holdsAt(meeting(fcgt_3,fcgt_4),371). :- not holdsAt(meeting(fcgt_4,fcgt_3),371).
:- not holdsAt(meeting(fcgt_3,fcgt_4),372). :- not holdsAt(meeting(fcgt_4,fcgt_3),372).
:- not holdsAt(meeting(fcgt_3,fcgt_4),373). :- not holdsAt(meeting(fcgt_4,fcgt_3),373).
:- not holdsAt(meeting(fcgt_3,fcgt_4),374). :- not holdsAt(meeting(fcgt_4,fcgt_3),374).
:- not holdsAt(meeting(fcgt_3,fcgt_4),375). :- not holdsAt(meeting(fcgt_4,fcgt_3),375).
:- not holdsAt(meeting(fcgt_3,fcgt_4),376). :- not holdsAt(meeting(fcgt_4,fcgt_3),376).
:- not holdsAt(meeting(fcgt_3,fcgt_4),377). :- not holdsAt(meeting(fcgt_4,fcgt_3),377).
:- not holdsAt(meeting(fcgt_3,fcgt_4),378). :- not holdsAt(meeting(fcgt_4,fcgt_3),378).
:- not holdsAt(meeting(fcgt_3,fcgt_4),379). :- not holdsAt(meeting(fcgt_4,fcgt_3),379).
:- not holdsAt(meeting(fcgt_3,fcgt_4),380). :- not holdsAt(meeting(fcgt_4,fcgt_3),380).
:- not holdsAt(meeting(fcgt_3,fcgt_4),381). :- not holdsAt(meeting(fcgt_4,fcgt_3),381).
:- not holdsAt(meeting(fcgt_3,fcgt_4),382). :- not holdsAt(meeting(fcgt_4,fcgt_3),382).
:- not holdsAt(meeting(fcgt_3,fcgt_4),383). :- not holdsAt(meeting(fcgt_4,fcgt_3),383).
:- not holdsAt(meeting(fcgt_3,fcgt_4),384). :- not holdsAt(meeting(fcgt_4,fcgt_3),384).
:- holdsAt(meeting(fcgt_3,fcgt_4),385). :- holdsAt(meeting(fcgt_4,fcgt_3),385).
:- simpleEvent(T,P,walking), T>360 , T<384.
:- simpleEvent(T,P,running), T>360 , T<384.

#show simpleEvent/3.