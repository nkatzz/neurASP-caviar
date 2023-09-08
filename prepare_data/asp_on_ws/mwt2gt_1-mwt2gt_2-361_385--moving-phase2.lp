

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

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,143,361). coords(mwt2gt_1,131,77,361).  visible(mwt2gt_1,361). orientation(mwt2gt_2,143,361). coords(mwt2gt_2,141,88,361).  visible(mwt2gt_2,361).
orientation(mwt2gt_1,143,362). coords(mwt2gt_1,130,78,362).  visible(mwt2gt_1,362). orientation(mwt2gt_2,144,362). coords(mwt2gt_2,140,89,362).  visible(mwt2gt_2,362).
orientation(mwt2gt_1,143,363). coords(mwt2gt_1,129,77,363).  visible(mwt2gt_1,363). orientation(mwt2gt_2,144,363). coords(mwt2gt_2,140,90,363).  visible(mwt2gt_2,363).
orientation(mwt2gt_1,143,364). coords(mwt2gt_1,129,78,364).  visible(mwt2gt_1,364). orientation(mwt2gt_2,144,364). coords(mwt2gt_2,139,92,364).  visible(mwt2gt_2,364).
orientation(mwt2gt_1,143,365). coords(mwt2gt_1,128,78,365).  visible(mwt2gt_1,365). orientation(mwt2gt_2,144,365). coords(mwt2gt_2,139,92,365).  visible(mwt2gt_2,365).
orientation(mwt2gt_1,143,366). coords(mwt2gt_1,127,78,366).  visible(mwt2gt_1,366). orientation(mwt2gt_2,144,366). coords(mwt2gt_2,138,93,366).  visible(mwt2gt_2,366).
orientation(mwt2gt_1,143,367). coords(mwt2gt_1,127,78,367).  visible(mwt2gt_1,367). orientation(mwt2gt_2,144,367). coords(mwt2gt_2,137,93,367).  visible(mwt2gt_2,367).
orientation(mwt2gt_1,143,368). coords(mwt2gt_1,127,79,368).  visible(mwt2gt_1,368). orientation(mwt2gt_2,144,368). coords(mwt2gt_2,137,95,368).  visible(mwt2gt_2,368).
orientation(mwt2gt_1,143,369). coords(mwt2gt_1,127,80,369).  visible(mwt2gt_1,369). orientation(mwt2gt_2,144,369). coords(mwt2gt_2,135,95,369).  visible(mwt2gt_2,369).
orientation(mwt2gt_1,145,370). coords(mwt2gt_1,126,80,370).  visible(mwt2gt_1,370). orientation(mwt2gt_2,144,370). coords(mwt2gt_2,133,95,370).  visible(mwt2gt_2,370).
orientation(mwt2gt_1,149,371). coords(mwt2gt_1,126,80,371).  visible(mwt2gt_1,371). orientation(mwt2gt_2,144,371). coords(mwt2gt_2,131,95,371).  visible(mwt2gt_2,371).
orientation(mwt2gt_1,149,372). coords(mwt2gt_1,126,82,372).  visible(mwt2gt_1,372). orientation(mwt2gt_2,151,372). coords(mwt2gt_2,129,96,372).  visible(mwt2gt_2,372).
orientation(mwt2gt_1,151,373). coords(mwt2gt_1,125,83,373).  visible(mwt2gt_1,373). orientation(mwt2gt_2,151,373). coords(mwt2gt_2,128,96,373).  visible(mwt2gt_2,373).
orientation(mwt2gt_1,153,374). coords(mwt2gt_1,124,83,374).  visible(mwt2gt_1,374). orientation(mwt2gt_2,151,374). coords(mwt2gt_2,128,96,374).  visible(mwt2gt_2,374).
orientation(mwt2gt_1,153,375). coords(mwt2gt_1,123,84,375).  visible(mwt2gt_1,375). orientation(mwt2gt_2,151,375). coords(mwt2gt_2,127,97,375).  visible(mwt2gt_2,375).
orientation(mwt2gt_1,153,376). coords(mwt2gt_1,122,84,376).  visible(mwt2gt_1,376). orientation(mwt2gt_2,151,376). coords(mwt2gt_2,127,97,376).  visible(mwt2gt_2,376).
orientation(mwt2gt_1,153,377). coords(mwt2gt_1,121,84,377).  visible(mwt2gt_1,377). orientation(mwt2gt_2,151,377). coords(mwt2gt_2,127,98,377).  visible(mwt2gt_2,377).
orientation(mwt2gt_1,153,378). coords(mwt2gt_1,119,84,378).  visible(mwt2gt_1,378). orientation(mwt2gt_2,151,378). coords(mwt2gt_2,126,99,378).  visible(mwt2gt_2,378).
orientation(mwt2gt_1,153,379). coords(mwt2gt_1,119,84,379).  visible(mwt2gt_1,379). orientation(mwt2gt_2,151,379). coords(mwt2gt_2,126,99,379).  visible(mwt2gt_2,379).
orientation(mwt2gt_1,153,380). coords(mwt2gt_1,119,85,380).  visible(mwt2gt_1,380). orientation(mwt2gt_2,151,380). coords(mwt2gt_2,124,101,380).  visible(mwt2gt_2,380).
orientation(mwt2gt_1,153,381). coords(mwt2gt_1,119,85,381).  visible(mwt2gt_1,381). orientation(mwt2gt_2,151,381). coords(mwt2gt_2,123,101,381).  visible(mwt2gt_2,381).
orientation(mwt2gt_1,153,382). coords(mwt2gt_1,118,85,382).  visible(mwt2gt_1,382). orientation(mwt2gt_2,151,382). coords(mwt2gt_2,121,101,382).  visible(mwt2gt_2,382).
orientation(mwt2gt_1,153,383). coords(mwt2gt_1,118,85,383).  visible(mwt2gt_1,383). orientation(mwt2gt_2,151,383). coords(mwt2gt_2,118,101,383).  visible(mwt2gt_2,383).
orientation(mwt2gt_1,153,384). coords(mwt2gt_1,118,85,384).  visible(mwt2gt_1,384). orientation(mwt2gt_2,151,384). coords(mwt2gt_2,117,102,384).  visible(mwt2gt_2,384).
orientation(mwt2gt_1,153,385). coords(mwt2gt_1,118,85,385).  visible(mwt2gt_1,385). orientation(mwt2gt_2,151,385). coords(mwt2gt_2,115,104,385).  visible(mwt2gt_2,385).
orientation(mwt2gt_1,153,386). coords(mwt2gt_1,118,85,386).  visible(mwt2gt_1,386). orientation(mwt2gt_2,151,386). coords(mwt2gt_2,115,104,386).  visible(mwt2gt_2,386).
happensAt(disappear(mwt2gt_1),385). happensAt(disappear(mwt2gt_2),385).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=361..385.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=361..385.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),361). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),361).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),362). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),362).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),363). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),363).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),364). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),364).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),365). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),365).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),366). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),366).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),367). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),367).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),368). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),368).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),369). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),369).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),370). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),370).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),371). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),371).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),372). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),372).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),373). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),373).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),374). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),374).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),375). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),375).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),376). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),376).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),377). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),377).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),378). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),378).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),379). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),379).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),380). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),380).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),381). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),381).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),382). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),382).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),383). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),383).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),384). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),384).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),385). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),385).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),386). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),386).
:- simpleEvent(T,P,inactive), T>361 , T<385.
:- simpleEvent(T,P,running), T>361 , T<385.

#show simpleEvent/3.