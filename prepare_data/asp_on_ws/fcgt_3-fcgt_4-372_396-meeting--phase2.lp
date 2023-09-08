

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
orientation(fcgt_3,164,385). coords(fcgt_3,99,188,385).  visible(fcgt_3,385). orientation(fcgt_4,166,385). coords(fcgt_4,82,206,385).  visible(fcgt_4,385).
orientation(fcgt_3,164,386). coords(fcgt_3,99,188,386).  visible(fcgt_3,386). orientation(fcgt_4,166,386). coords(fcgt_4,82,206,386).  visible(fcgt_4,386).
orientation(fcgt_3,164,387). coords(fcgt_3,99,188,387).  visible(fcgt_3,387). orientation(fcgt_4,166,387). coords(fcgt_4,82,206,387).  visible(fcgt_4,387).
orientation(fcgt_3,164,388). coords(fcgt_3,99,189,388).  visible(fcgt_3,388). orientation(fcgt_4,166,388). coords(fcgt_4,82,206,388).  visible(fcgt_4,388).
orientation(fcgt_3,164,389). coords(fcgt_3,99,189,389).  visible(fcgt_3,389). orientation(fcgt_4,166,389). coords(fcgt_4,82,206,389).  visible(fcgt_4,389).
orientation(fcgt_3,164,390). coords(fcgt_3,100,190,390).  visible(fcgt_3,390). orientation(fcgt_4,166,390). coords(fcgt_4,82,206,390).  visible(fcgt_4,390).
orientation(fcgt_3,164,391). coords(fcgt_3,100,190,391).  visible(fcgt_3,391). orientation(fcgt_4,166,391). coords(fcgt_4,82,206,391).  visible(fcgt_4,391).
orientation(fcgt_3,164,392). coords(fcgt_3,100,190,392).  visible(fcgt_3,392). orientation(fcgt_4,166,392). coords(fcgt_4,82,206,392).  visible(fcgt_4,392).
orientation(fcgt_3,164,393). coords(fcgt_3,100,190,393).  visible(fcgt_3,393). orientation(fcgt_4,166,393). coords(fcgt_4,82,206,393).  visible(fcgt_4,393).
orientation(fcgt_3,164,394). coords(fcgt_3,100,191,394).  visible(fcgt_3,394). orientation(fcgt_4,166,394). coords(fcgt_4,82,206,394).  visible(fcgt_4,394).
orientation(fcgt_3,164,395). coords(fcgt_3,100,192,395).  visible(fcgt_3,395). orientation(fcgt_4,166,395). coords(fcgt_4,82,206,395).  visible(fcgt_4,395).
orientation(fcgt_3,164,396). coords(fcgt_3,100,193,396).  visible(fcgt_3,396). orientation(fcgt_4,166,396). coords(fcgt_4,82,206,396).  visible(fcgt_4,396).
orientation(fcgt_3,164,397). coords(fcgt_3,100,193,397).  visible(fcgt_3,397). orientation(fcgt_4,166,397). coords(fcgt_4,82,206,397).  visible(fcgt_4,397).
happensAt(disappear(fcgt_3),396). happensAt(disappear(fcgt_4),396).
1{simpleEvent(T,fcgt_3,active); simpleEvent(T,fcgt_3,inactive); simpleEvent(T,fcgt_3,walking); simpleEvent(T,fcgt_3,running)}1 :- T=372..396.
1{simpleEvent(T,fcgt_4,active); simpleEvent(T,fcgt_4,inactive); simpleEvent(T,fcgt_4,walking); simpleEvent(T,fcgt_4,running)}1 :- T=372..396.
:- holdsAt(meeting(fcgt_3,fcgt_4),372). :- holdsAt(meeting(fcgt_4,fcgt_3),372).
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
:- not holdsAt(meeting(fcgt_3,fcgt_4),385). :- not holdsAt(meeting(fcgt_4,fcgt_3),385).
:- not holdsAt(meeting(fcgt_3,fcgt_4),386). :- not holdsAt(meeting(fcgt_4,fcgt_3),386).
:- not holdsAt(meeting(fcgt_3,fcgt_4),387). :- not holdsAt(meeting(fcgt_4,fcgt_3),387).
:- not holdsAt(meeting(fcgt_3,fcgt_4),388). :- not holdsAt(meeting(fcgt_4,fcgt_3),388).
:- not holdsAt(meeting(fcgt_3,fcgt_4),389). :- not holdsAt(meeting(fcgt_4,fcgt_3),389).
:- not holdsAt(meeting(fcgt_3,fcgt_4),390). :- not holdsAt(meeting(fcgt_4,fcgt_3),390).
:- not holdsAt(meeting(fcgt_3,fcgt_4),391). :- not holdsAt(meeting(fcgt_4,fcgt_3),391).
:- not holdsAt(meeting(fcgt_3,fcgt_4),392). :- not holdsAt(meeting(fcgt_4,fcgt_3),392).
:- not holdsAt(meeting(fcgt_3,fcgt_4),393). :- not holdsAt(meeting(fcgt_4,fcgt_3),393).
:- not holdsAt(meeting(fcgt_3,fcgt_4),394). :- not holdsAt(meeting(fcgt_4,fcgt_3),394).
:- not holdsAt(meeting(fcgt_3,fcgt_4),395). :- not holdsAt(meeting(fcgt_4,fcgt_3),395).
:- not holdsAt(meeting(fcgt_3,fcgt_4),396). :- not holdsAt(meeting(fcgt_4,fcgt_3),396).
:- holdsAt(meeting(fcgt_3,fcgt_4),397). :- holdsAt(meeting(fcgt_4,fcgt_3),397).
:- simpleEvent(T,P,walking), T>372 , T<396.
:- simpleEvent(T,P,running), T>372 , T<396.

#show simpleEvent/3.