

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
orientation(fcgt_3,164,324). coords(fcgt_3,98,187,324).  visible(fcgt_3,324). orientation(fcgt_4,166,324). coords(fcgt_4,82,206,324).  visible(fcgt_4,324).
orientation(fcgt_3,164,325). coords(fcgt_3,98,187,325).  visible(fcgt_3,325). orientation(fcgt_4,166,325). coords(fcgt_4,82,206,325).  visible(fcgt_4,325).
orientation(fcgt_3,164,326). coords(fcgt_3,98,187,326).  visible(fcgt_3,326). orientation(fcgt_4,166,326). coords(fcgt_4,82,206,326).  visible(fcgt_4,326).
orientation(fcgt_3,164,327). coords(fcgt_3,98,187,327).  visible(fcgt_3,327). orientation(fcgt_4,166,327). coords(fcgt_4,82,206,327).  visible(fcgt_4,327).
orientation(fcgt_3,164,328). coords(fcgt_3,98,187,328).  visible(fcgt_3,328). orientation(fcgt_4,166,328). coords(fcgt_4,82,206,328).  visible(fcgt_4,328).
orientation(fcgt_3,164,329). coords(fcgt_3,98,187,329).  visible(fcgt_3,329). orientation(fcgt_4,166,329). coords(fcgt_4,82,206,329).  visible(fcgt_4,329).
orientation(fcgt_3,164,330). coords(fcgt_3,98,187,330).  visible(fcgt_3,330). orientation(fcgt_4,166,330). coords(fcgt_4,82,206,330).  visible(fcgt_4,330).
orientation(fcgt_3,164,331). coords(fcgt_3,98,187,331).  visible(fcgt_3,331). orientation(fcgt_4,166,331). coords(fcgt_4,82,206,331).  visible(fcgt_4,331).
orientation(fcgt_3,164,332). coords(fcgt_3,98,187,332).  visible(fcgt_3,332). orientation(fcgt_4,166,332). coords(fcgt_4,82,206,332).  visible(fcgt_4,332).
orientation(fcgt_3,164,333). coords(fcgt_3,98,187,333).  visible(fcgt_3,333). orientation(fcgt_4,166,333). coords(fcgt_4,82,206,333).  visible(fcgt_4,333).
orientation(fcgt_3,164,334). coords(fcgt_3,98,187,334).  visible(fcgt_3,334). orientation(fcgt_4,166,334). coords(fcgt_4,82,206,334).  visible(fcgt_4,334).
orientation(fcgt_3,164,335). coords(fcgt_3,98,187,335).  visible(fcgt_3,335). orientation(fcgt_4,166,335). coords(fcgt_4,82,206,335).  visible(fcgt_4,335).
orientation(fcgt_3,164,336). coords(fcgt_3,98,187,336).  visible(fcgt_3,336). orientation(fcgt_4,166,336). coords(fcgt_4,82,206,336).  visible(fcgt_4,336).
orientation(fcgt_3,164,337). coords(fcgt_3,98,187,337).  visible(fcgt_3,337). orientation(fcgt_4,166,337). coords(fcgt_4,82,206,337).  visible(fcgt_4,337).
orientation(fcgt_3,164,338). coords(fcgt_3,98,187,338).  visible(fcgt_3,338). orientation(fcgt_4,166,338). coords(fcgt_4,82,206,338).  visible(fcgt_4,338).
orientation(fcgt_3,164,339). coords(fcgt_3,98,187,339).  visible(fcgt_3,339). orientation(fcgt_4,166,339). coords(fcgt_4,82,206,339).  visible(fcgt_4,339).
orientation(fcgt_3,164,340). coords(fcgt_3,98,187,340).  visible(fcgt_3,340). orientation(fcgt_4,166,340). coords(fcgt_4,82,206,340).  visible(fcgt_4,340).
orientation(fcgt_3,164,341). coords(fcgt_3,98,187,341).  visible(fcgt_3,341). orientation(fcgt_4,166,341). coords(fcgt_4,82,206,341).  visible(fcgt_4,341).
orientation(fcgt_3,164,342). coords(fcgt_3,98,187,342).  visible(fcgt_3,342). orientation(fcgt_4,166,342). coords(fcgt_4,82,206,342).  visible(fcgt_4,342).
orientation(fcgt_3,164,343). coords(fcgt_3,98,187,343).  visible(fcgt_3,343). orientation(fcgt_4,166,343). coords(fcgt_4,82,206,343).  visible(fcgt_4,343).
orientation(fcgt_3,164,344). coords(fcgt_3,98,187,344).  visible(fcgt_3,344). orientation(fcgt_4,166,344). coords(fcgt_4,82,206,344).  visible(fcgt_4,344).
orientation(fcgt_3,164,345). coords(fcgt_3,98,187,345).  visible(fcgt_3,345). orientation(fcgt_4,166,345). coords(fcgt_4,82,206,345).  visible(fcgt_4,345).
orientation(fcgt_3,164,346). coords(fcgt_3,98,187,346).  visible(fcgt_3,346). orientation(fcgt_4,166,346). coords(fcgt_4,82,206,346).  visible(fcgt_4,346).
orientation(fcgt_3,164,347). coords(fcgt_3,98,187,347).  visible(fcgt_3,347). orientation(fcgt_4,166,347). coords(fcgt_4,82,206,347).  visible(fcgt_4,347).
orientation(fcgt_3,164,348). coords(fcgt_3,98,187,348).  visible(fcgt_3,348). orientation(fcgt_4,166,348). coords(fcgt_4,82,206,348).  visible(fcgt_4,348).
orientation(fcgt_3,164,349). coords(fcgt_3,98,187,349).  visible(fcgt_3,349). orientation(fcgt_4,166,349). coords(fcgt_4,82,206,349).  visible(fcgt_4,349).
happensAt(disappear(fcgt_3),348). happensAt(disappear(fcgt_4),348).
1{simpleEvent(T,fcgt_3,active); simpleEvent(T,fcgt_3,inactive); simpleEvent(T,fcgt_3,walking); simpleEvent(T,fcgt_3,running)}1 :- T=324..348.
1{simpleEvent(T,fcgt_4,active); simpleEvent(T,fcgt_4,inactive); simpleEvent(T,fcgt_4,walking); simpleEvent(T,fcgt_4,running)}1 :- T=324..348.
:- holdsAt(meeting(fcgt_3,fcgt_4),324). :- holdsAt(meeting(fcgt_4,fcgt_3),324).
:- not holdsAt(meeting(fcgt_3,fcgt_4),325). :- not holdsAt(meeting(fcgt_4,fcgt_3),325).
:- not holdsAt(meeting(fcgt_3,fcgt_4),326). :- not holdsAt(meeting(fcgt_4,fcgt_3),326).
:- not holdsAt(meeting(fcgt_3,fcgt_4),327). :- not holdsAt(meeting(fcgt_4,fcgt_3),327).
:- not holdsAt(meeting(fcgt_3,fcgt_4),328). :- not holdsAt(meeting(fcgt_4,fcgt_3),328).
:- not holdsAt(meeting(fcgt_3,fcgt_4),329). :- not holdsAt(meeting(fcgt_4,fcgt_3),329).
:- not holdsAt(meeting(fcgt_3,fcgt_4),330). :- not holdsAt(meeting(fcgt_4,fcgt_3),330).
:- not holdsAt(meeting(fcgt_3,fcgt_4),331). :- not holdsAt(meeting(fcgt_4,fcgt_3),331).
:- not holdsAt(meeting(fcgt_3,fcgt_4),332). :- not holdsAt(meeting(fcgt_4,fcgt_3),332).
:- not holdsAt(meeting(fcgt_3,fcgt_4),333). :- not holdsAt(meeting(fcgt_4,fcgt_3),333).
:- not holdsAt(meeting(fcgt_3,fcgt_4),334). :- not holdsAt(meeting(fcgt_4,fcgt_3),334).
:- not holdsAt(meeting(fcgt_3,fcgt_4),335). :- not holdsAt(meeting(fcgt_4,fcgt_3),335).
:- not holdsAt(meeting(fcgt_3,fcgt_4),336). :- not holdsAt(meeting(fcgt_4,fcgt_3),336).
:- not holdsAt(meeting(fcgt_3,fcgt_4),337). :- not holdsAt(meeting(fcgt_4,fcgt_3),337).
:- not holdsAt(meeting(fcgt_3,fcgt_4),338). :- not holdsAt(meeting(fcgt_4,fcgt_3),338).
:- not holdsAt(meeting(fcgt_3,fcgt_4),339). :- not holdsAt(meeting(fcgt_4,fcgt_3),339).
:- not holdsAt(meeting(fcgt_3,fcgt_4),340). :- not holdsAt(meeting(fcgt_4,fcgt_3),340).
:- not holdsAt(meeting(fcgt_3,fcgt_4),341). :- not holdsAt(meeting(fcgt_4,fcgt_3),341).
:- not holdsAt(meeting(fcgt_3,fcgt_4),342). :- not holdsAt(meeting(fcgt_4,fcgt_3),342).
:- not holdsAt(meeting(fcgt_3,fcgt_4),343). :- not holdsAt(meeting(fcgt_4,fcgt_3),343).
:- not holdsAt(meeting(fcgt_3,fcgt_4),344). :- not holdsAt(meeting(fcgt_4,fcgt_3),344).
:- not holdsAt(meeting(fcgt_3,fcgt_4),345). :- not holdsAt(meeting(fcgt_4,fcgt_3),345).
:- not holdsAt(meeting(fcgt_3,fcgt_4),346). :- not holdsAt(meeting(fcgt_4,fcgt_3),346).
:- not holdsAt(meeting(fcgt_3,fcgt_4),347). :- not holdsAt(meeting(fcgt_4,fcgt_3),347).
:- not holdsAt(meeting(fcgt_3,fcgt_4),348). :- not holdsAt(meeting(fcgt_4,fcgt_3),348).
:- holdsAt(meeting(fcgt_3,fcgt_4),349). :- holdsAt(meeting(fcgt_4,fcgt_3),349).
:- simpleEvent(T,P,walking), T>324 , T<348.
:- simpleEvent(T,P,running), T>324 , T<348.

#show simpleEvent/3.