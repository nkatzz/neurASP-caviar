

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

	person(fcgt_1;fcgt_2).
orientation(fcgt_1,156,328). coords(fcgt_1,103,137,328).  visible(fcgt_1,328). orientation(fcgt_2,146,328). coords(fcgt_2,91,127,328).  visible(fcgt_2,328).
orientation(fcgt_1,160,329). coords(fcgt_1,103,136,329).  visible(fcgt_1,329). orientation(fcgt_2,146,329). coords(fcgt_2,91,126,329).  visible(fcgt_2,329).
orientation(fcgt_1,163,330). coords(fcgt_1,103,135,330).  visible(fcgt_1,330). orientation(fcgt_2,146,330). coords(fcgt_2,91,125,330).  visible(fcgt_2,330).
orientation(fcgt_1,160,331). coords(fcgt_1,101,134,331).  visible(fcgt_1,331). orientation(fcgt_2,146,331). coords(fcgt_2,91,124,331).  visible(fcgt_2,331).
orientation(fcgt_1,160,332). coords(fcgt_1,99,134,332).  visible(fcgt_1,332). orientation(fcgt_2,146,332). coords(fcgt_2,91,124,332).  visible(fcgt_2,332).
orientation(fcgt_1,161,333). coords(fcgt_1,99,135,333).  visible(fcgt_1,333). orientation(fcgt_2,146,333). coords(fcgt_2,92,124,333).  visible(fcgt_2,333).
orientation(fcgt_1,158,334). coords(fcgt_1,98,135,334).  visible(fcgt_1,334). orientation(fcgt_2,146,334). coords(fcgt_2,93,123,334).  visible(fcgt_2,334).
orientation(fcgt_1,153,335). coords(fcgt_1,97,134,335).  visible(fcgt_1,335). orientation(fcgt_2,146,335). coords(fcgt_2,94,123,335).  visible(fcgt_2,335).
orientation(fcgt_1,151,336). coords(fcgt_1,97,134,336).  visible(fcgt_1,336). orientation(fcgt_2,155,336). coords(fcgt_2,94,121,336).  visible(fcgt_2,336).
orientation(fcgt_1,147,337). coords(fcgt_1,96,133,337).  visible(fcgt_1,337). orientation(fcgt_2,155,337). coords(fcgt_2,94,120,337).  visible(fcgt_2,337).
orientation(fcgt_1,147,338). coords(fcgt_1,96,132,338).  visible(fcgt_1,338). orientation(fcgt_2,155,338). coords(fcgt_2,94,120,338).  visible(fcgt_2,338).
orientation(fcgt_1,147,339). coords(fcgt_1,96,130,339).  visible(fcgt_1,339). orientation(fcgt_2,155,339). coords(fcgt_2,94,119,339).  visible(fcgt_2,339).
orientation(fcgt_1,147,340). coords(fcgt_1,96,129,340).  visible(fcgt_1,340). orientation(fcgt_2,155,340). coords(fcgt_2,94,117,340).  visible(fcgt_2,340).
orientation(fcgt_1,150,341). coords(fcgt_1,96,129,341).  visible(fcgt_1,341). orientation(fcgt_2,155,341). coords(fcgt_2,94,117,341).  visible(fcgt_2,341).
orientation(fcgt_1,148,342). coords(fcgt_1,96,129,342).  visible(fcgt_1,342). orientation(fcgt_2,155,342). coords(fcgt_2,94,116,342).  visible(fcgt_2,342).
orientation(fcgt_1,146,343). coords(fcgt_1,96,130,343).  visible(fcgt_1,343). orientation(fcgt_2,155,343). coords(fcgt_2,94,115,343).  visible(fcgt_2,343).
orientation(fcgt_1,146,344). coords(fcgt_1,96,130,344).  visible(fcgt_1,344). orientation(fcgt_2,155,344). coords(fcgt_2,94,115,344).  visible(fcgt_2,344).
orientation(fcgt_1,146,345). coords(fcgt_1,97,129,345).  visible(fcgt_1,345). orientation(fcgt_2,155,345). coords(fcgt_2,94,114,345).  visible(fcgt_2,345).
orientation(fcgt_1,146,346). coords(fcgt_1,97,129,346).  visible(fcgt_1,346). orientation(fcgt_2,155,346). coords(fcgt_2,93,114,346).  visible(fcgt_2,346).
orientation(fcgt_1,141,347). coords(fcgt_1,97,127,347).  visible(fcgt_1,347). orientation(fcgt_2,155,347). coords(fcgt_2,94,114,347).  visible(fcgt_2,347).
orientation(fcgt_1,141,348). coords(fcgt_1,97,126,348).  visible(fcgt_1,348). orientation(fcgt_2,155,348). coords(fcgt_2,94,114,348).  visible(fcgt_2,348).
orientation(fcgt_1,141,349). coords(fcgt_1,97,124,349).  visible(fcgt_1,349). orientation(fcgt_2,155,349). coords(fcgt_2,93,113,349).  visible(fcgt_2,349).
orientation(fcgt_1,141,350). coords(fcgt_1,97,123,350).  visible(fcgt_1,350). orientation(fcgt_2,154,350). coords(fcgt_2,93,112,350).  visible(fcgt_2,350).
orientation(fcgt_1,141,351). coords(fcgt_1,96,122,351).  visible(fcgt_1,351). orientation(fcgt_2,149,351). coords(fcgt_2,92,111,351).  visible(fcgt_2,351).
orientation(fcgt_1,141,352). coords(fcgt_1,96,123,352).  visible(fcgt_1,352). orientation(fcgt_2,147,352). coords(fcgt_2,92,111,352).  visible(fcgt_2,352).
orientation(fcgt_1,141,353). coords(fcgt_1,96,123,353).  visible(fcgt_1,353). orientation(fcgt_2,147,353). coords(fcgt_2,92,111,353).  visible(fcgt_2,353).
happensAt(disappear(fcgt_1),352). happensAt(disappear(fcgt_2),352).
1{simpleEvent(T,fcgt_1,active); simpleEvent(T,fcgt_1,inactive); simpleEvent(T,fcgt_1,walking); simpleEvent(T,fcgt_1,running)}1 :- T=328..352.
1{simpleEvent(T,fcgt_2,active); simpleEvent(T,fcgt_2,inactive); simpleEvent(T,fcgt_2,walking); simpleEvent(T,fcgt_2,running)}1 :- T=328..352.
:- holdsAt(meeting(fcgt_1,fcgt_2),328). :- holdsAt(meeting(fcgt_2,fcgt_1),328).
:- not holdsAt(meeting(fcgt_1,fcgt_2),329). :- not holdsAt(meeting(fcgt_2,fcgt_1),329).
:- not holdsAt(meeting(fcgt_1,fcgt_2),330). :- not holdsAt(meeting(fcgt_2,fcgt_1),330).
:- not holdsAt(meeting(fcgt_1,fcgt_2),331). :- not holdsAt(meeting(fcgt_2,fcgt_1),331).
:- not holdsAt(meeting(fcgt_1,fcgt_2),332). :- not holdsAt(meeting(fcgt_2,fcgt_1),332).
:- not holdsAt(meeting(fcgt_1,fcgt_2),333). :- not holdsAt(meeting(fcgt_2,fcgt_1),333).
:- not holdsAt(meeting(fcgt_1,fcgt_2),334). :- not holdsAt(meeting(fcgt_2,fcgt_1),334).
:- not holdsAt(meeting(fcgt_1,fcgt_2),335). :- not holdsAt(meeting(fcgt_2,fcgt_1),335).
:- not holdsAt(meeting(fcgt_1,fcgt_2),336). :- not holdsAt(meeting(fcgt_2,fcgt_1),336).
:- not holdsAt(meeting(fcgt_1,fcgt_2),337). :- not holdsAt(meeting(fcgt_2,fcgt_1),337).
:- not holdsAt(meeting(fcgt_1,fcgt_2),338). :- not holdsAt(meeting(fcgt_2,fcgt_1),338).
:- not holdsAt(meeting(fcgt_1,fcgt_2),339). :- not holdsAt(meeting(fcgt_2,fcgt_1),339).
:- not holdsAt(meeting(fcgt_1,fcgt_2),340). :- not holdsAt(meeting(fcgt_2,fcgt_1),340).
:- not holdsAt(meeting(fcgt_1,fcgt_2),341). :- not holdsAt(meeting(fcgt_2,fcgt_1),341).
:- not holdsAt(meeting(fcgt_1,fcgt_2),342). :- not holdsAt(meeting(fcgt_2,fcgt_1),342).
:- not holdsAt(meeting(fcgt_1,fcgt_2),343). :- not holdsAt(meeting(fcgt_2,fcgt_1),343).
:- not holdsAt(meeting(fcgt_1,fcgt_2),344). :- not holdsAt(meeting(fcgt_2,fcgt_1),344).
:- not holdsAt(meeting(fcgt_1,fcgt_2),345). :- not holdsAt(meeting(fcgt_2,fcgt_1),345).
:- not holdsAt(meeting(fcgt_1,fcgt_2),346). :- not holdsAt(meeting(fcgt_2,fcgt_1),346).
:- not holdsAt(meeting(fcgt_1,fcgt_2),347). :- not holdsAt(meeting(fcgt_2,fcgt_1),347).
:- not holdsAt(meeting(fcgt_1,fcgt_2),348). :- not holdsAt(meeting(fcgt_2,fcgt_1),348).
:- not holdsAt(meeting(fcgt_1,fcgt_2),349). :- not holdsAt(meeting(fcgt_2,fcgt_1),349).
:- not holdsAt(meeting(fcgt_1,fcgt_2),350). :- not holdsAt(meeting(fcgt_2,fcgt_1),350).
:- not holdsAt(meeting(fcgt_1,fcgt_2),351). :- not holdsAt(meeting(fcgt_2,fcgt_1),351).
:- not holdsAt(meeting(fcgt_1,fcgt_2),352). :- not holdsAt(meeting(fcgt_2,fcgt_1),352).
:- holdsAt(meeting(fcgt_1,fcgt_2),353). :- holdsAt(meeting(fcgt_2,fcgt_1),353).
:- simpleEvent(T,P,walking), T>328 , T<352.
:- simpleEvent(T,P,running), T>328 , T<352.

#show simpleEvent/3.