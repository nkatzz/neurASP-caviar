

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
orientation(mwt2gt_1,143,325). coords(mwt2gt_1,138,75,325).  visible(mwt2gt_1,325). orientation(mwt2gt_2,145,325). coords(mwt2gt_2,142,87,325).  visible(mwt2gt_2,325).
orientation(mwt2gt_1,143,326). coords(mwt2gt_1,138,75,326).  visible(mwt2gt_1,326). orientation(mwt2gt_2,144,326). coords(mwt2gt_2,142,87,326).  visible(mwt2gt_2,326).
orientation(mwt2gt_1,143,327). coords(mwt2gt_1,138,75,327).  visible(mwt2gt_1,327). orientation(mwt2gt_2,143,327). coords(mwt2gt_2,143,87,327).  visible(mwt2gt_2,327).
orientation(mwt2gt_1,143,328). coords(mwt2gt_1,138,76,328).  visible(mwt2gt_1,328). orientation(mwt2gt_2,145,328). coords(mwt2gt_2,143,87,328).  visible(mwt2gt_2,328).
orientation(mwt2gt_1,143,329). coords(mwt2gt_1,138,76,329).  visible(mwt2gt_1,329). orientation(mwt2gt_2,145,329). coords(mwt2gt_2,143,87,329).  visible(mwt2gt_2,329).
orientation(mwt2gt_1,143,330). coords(mwt2gt_1,138,76,330).  visible(mwt2gt_1,330). orientation(mwt2gt_2,145,330). coords(mwt2gt_2,143,87,330).  visible(mwt2gt_2,330).
orientation(mwt2gt_1,143,331). coords(mwt2gt_1,138,76,331).  visible(mwt2gt_1,331). orientation(mwt2gt_2,145,331). coords(mwt2gt_2,143,87,331).  visible(mwt2gt_2,331).
orientation(mwt2gt_1,143,332). coords(mwt2gt_1,138,76,332).  visible(mwt2gt_1,332). orientation(mwt2gt_2,145,332). coords(mwt2gt_2,143,87,332).  visible(mwt2gt_2,332).
orientation(mwt2gt_1,143,333). coords(mwt2gt_1,137,76,333).  visible(mwt2gt_1,333). orientation(mwt2gt_2,145,333). coords(mwt2gt_2,143,87,333).  visible(mwt2gt_2,333).
orientation(mwt2gt_1,143,334). coords(mwt2gt_1,137,76,334).  visible(mwt2gt_1,334). orientation(mwt2gt_2,145,334). coords(mwt2gt_2,142,87,334).  visible(mwt2gt_2,334).
orientation(mwt2gt_1,143,335). coords(mwt2gt_1,137,76,335).  visible(mwt2gt_1,335). orientation(mwt2gt_2,145,335). coords(mwt2gt_2,142,87,335).  visible(mwt2gt_2,335).
orientation(mwt2gt_1,143,336). coords(mwt2gt_1,137,76,336).  visible(mwt2gt_1,336). orientation(mwt2gt_2,145,336). coords(mwt2gt_2,142,87,336).  visible(mwt2gt_2,336).
orientation(mwt2gt_1,143,337). coords(mwt2gt_1,137,76,337).  visible(mwt2gt_1,337). orientation(mwt2gt_2,145,337). coords(mwt2gt_2,142,87,337).  visible(mwt2gt_2,337).
orientation(mwt2gt_1,143,338). coords(mwt2gt_1,137,76,338).  visible(mwt2gt_1,338). orientation(mwt2gt_2,145,338). coords(mwt2gt_2,142,87,338).  visible(mwt2gt_2,338).
orientation(mwt2gt_1,143,339). coords(mwt2gt_1,137,76,339).  visible(mwt2gt_1,339). orientation(mwt2gt_2,145,339). coords(mwt2gt_2,142,87,339).  visible(mwt2gt_2,339).
orientation(mwt2gt_1,143,340). coords(mwt2gt_1,134,77,340).  visible(mwt2gt_1,340). orientation(mwt2gt_2,141,340). coords(mwt2gt_2,142,86,340).  visible(mwt2gt_2,340).
orientation(mwt2gt_1,143,341). coords(mwt2gt_1,134,77,341).  visible(mwt2gt_1,341). orientation(mwt2gt_2,141,341). coords(mwt2gt_2,142,86,341).  visible(mwt2gt_2,341).
orientation(mwt2gt_1,143,342). coords(mwt2gt_1,134,77,342).  visible(mwt2gt_1,342). orientation(mwt2gt_2,141,342). coords(mwt2gt_2,142,86,342).  visible(mwt2gt_2,342).
orientation(mwt2gt_1,143,343). coords(mwt2gt_1,134,77,343).  visible(mwt2gt_1,343). orientation(mwt2gt_2,141,343). coords(mwt2gt_2,142,86,343).  visible(mwt2gt_2,343).
orientation(mwt2gt_1,143,344). coords(mwt2gt_1,134,77,344).  visible(mwt2gt_1,344). orientation(mwt2gt_2,141,344). coords(mwt2gt_2,142,86,344).  visible(mwt2gt_2,344).
orientation(mwt2gt_1,143,345). coords(mwt2gt_1,134,77,345).  visible(mwt2gt_1,345). orientation(mwt2gt_2,141,345). coords(mwt2gt_2,142,86,345).  visible(mwt2gt_2,345).
orientation(mwt2gt_1,143,346). coords(mwt2gt_1,134,77,346).  visible(mwt2gt_1,346). orientation(mwt2gt_2,141,346). coords(mwt2gt_2,142,86,346).  visible(mwt2gt_2,346).
orientation(mwt2gt_1,143,347). coords(mwt2gt_1,134,77,347).  visible(mwt2gt_1,347). orientation(mwt2gt_2,141,347). coords(mwt2gt_2,142,86,347).  visible(mwt2gt_2,347).
orientation(mwt2gt_1,143,348). coords(mwt2gt_1,134,77,348).  visible(mwt2gt_1,348). orientation(mwt2gt_2,141,348). coords(mwt2gt_2,142,86,348).  visible(mwt2gt_2,348).
orientation(mwt2gt_1,143,349). coords(mwt2gt_1,134,77,349).  visible(mwt2gt_1,349). orientation(mwt2gt_2,141,349). coords(mwt2gt_2,142,86,349).  visible(mwt2gt_2,349).
orientation(mwt2gt_1,143,350). coords(mwt2gt_1,134,77,350).  visible(mwt2gt_1,350). orientation(mwt2gt_2,141,350). coords(mwt2gt_2,142,86,350).  visible(mwt2gt_2,350).
happensAt(disappear(mwt2gt_1),349). happensAt(disappear(mwt2gt_2),349).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=325..349.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=325..349.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),325). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),325).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),326). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),326).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),327). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),327).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),328). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),328).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),329). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),329).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),330). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),330).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),331). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),331).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),332). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),332).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),333). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),333).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),334). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),334).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),335). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),335).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),336). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),336).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),337). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),337).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),338). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),338).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),339). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),339).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),340). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),340).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),341). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),341).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),342). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),342).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),343). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),343).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),344). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),344).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),345). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),345).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),346). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),346).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),347). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),347).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),348). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),348).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),349). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),349).
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),350). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),350).
:- simpleEvent(T,P,walking), T>339 , T<349.
:- simpleEvent(T,P,running), T>339 , T<349.

#show simpleEvent/3.