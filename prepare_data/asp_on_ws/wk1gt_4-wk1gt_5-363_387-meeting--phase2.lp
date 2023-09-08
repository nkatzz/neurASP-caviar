

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
orientation(wk1gt_4,163,376). coords(wk1gt_4,52,176,376).  visible(wk1gt_4,376). orientation(wk1gt_5,162,376). coords(wk1gt_5,50,163,376).  visible(wk1gt_5,376).
orientation(wk1gt_4,163,377). coords(wk1gt_4,52,176,377).  visible(wk1gt_4,377). orientation(wk1gt_5,162,377). coords(wk1gt_5,51,163,377).  visible(wk1gt_5,377).
orientation(wk1gt_4,161,378). coords(wk1gt_4,52,176,378).  visible(wk1gt_4,378). orientation(wk1gt_5,162,378). coords(wk1gt_5,50,163,378).  visible(wk1gt_5,378).
orientation(wk1gt_4,154,379). coords(wk1gt_4,52,176,379).  visible(wk1gt_4,379). orientation(wk1gt_5,162,379). coords(wk1gt_5,50,163,379).  visible(wk1gt_5,379).
orientation(wk1gt_4,157,380). coords(wk1gt_4,52,176,380).  visible(wk1gt_4,380). orientation(wk1gt_5,162,380). coords(wk1gt_5,50,163,380).  visible(wk1gt_5,380).
orientation(wk1gt_4,159,381). coords(wk1gt_4,52,176,381).  visible(wk1gt_4,381). orientation(wk1gt_5,162,381). coords(wk1gt_5,50,163,381).  visible(wk1gt_5,381).
orientation(wk1gt_4,155,382). coords(wk1gt_4,52,176,382).  visible(wk1gt_4,382). orientation(wk1gt_5,162,382). coords(wk1gt_5,50,163,382).  visible(wk1gt_5,382).
orientation(wk1gt_4,159,383). coords(wk1gt_4,52,176,383).  visible(wk1gt_4,383). orientation(wk1gt_5,162,383). coords(wk1gt_5,50,163,383).  visible(wk1gt_5,383).
orientation(wk1gt_4,152,384). coords(wk1gt_4,52,175,384).  visible(wk1gt_4,384). orientation(wk1gt_5,162,384). coords(wk1gt_5,50,163,384).  visible(wk1gt_5,384).
orientation(wk1gt_4,154,385). coords(wk1gt_4,52,175,385).  visible(wk1gt_4,385). orientation(wk1gt_5,162,385). coords(wk1gt_5,50,163,385).  visible(wk1gt_5,385).
orientation(wk1gt_4,155,386). coords(wk1gt_4,52,174,386).  visible(wk1gt_4,386). orientation(wk1gt_5,162,386). coords(wk1gt_5,50,163,386).  visible(wk1gt_5,386).
orientation(wk1gt_4,156,387). coords(wk1gt_4,52,174,387).  visible(wk1gt_4,387). orientation(wk1gt_5,162,387). coords(wk1gt_5,50,163,387).  visible(wk1gt_5,387).
orientation(wk1gt_4,156,388). coords(wk1gt_4,52,174,388).  visible(wk1gt_4,388). orientation(wk1gt_5,162,388). coords(wk1gt_5,50,163,388).  visible(wk1gt_5,388).
happensAt(disappear(wk1gt_4),387). happensAt(disappear(wk1gt_5),387).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=363..387.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=363..387.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),363). :- holdsAt(meeting(wk1gt_5,wk1gt_4),363).
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
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),376). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),376).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),377). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),377).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),378). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),378).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),379). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),379).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),380). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),380).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),381). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),381).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),382). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),382).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),383). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),383).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),384). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),384).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),385). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),385).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),386). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),386).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),387). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),387).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),388). :- holdsAt(meeting(wk1gt_5,wk1gt_4),388).
:- simpleEvent(T,P,walking), T>363 , T<387.
:- simpleEvent(T,P,running), T>363 , T<387.

#show simpleEvent/3.