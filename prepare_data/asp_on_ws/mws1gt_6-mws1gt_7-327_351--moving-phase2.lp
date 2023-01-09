

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

	person(mws1gt_6;mws1gt_7).
orientation(mws1gt_6,146,327). coords(mws1gt_6,33,178,327).  visible(mws1gt_6,327). orientation(mws1gt_7,162,327). coords(mws1gt_7,38,199,327).  visible(mws1gt_7,327).
orientation(mws1gt_6,146,328). coords(mws1gt_6,33,178,328).  visible(mws1gt_6,328). orientation(mws1gt_7,167,328). coords(mws1gt_7,38,197,328).  visible(mws1gt_7,328).
orientation(mws1gt_6,146,329). coords(mws1gt_6,33,178,329).  visible(mws1gt_6,329). orientation(mws1gt_7,167,329). coords(mws1gt_7,39,197,329).  visible(mws1gt_7,329).
orientation(mws1gt_6,146,330). coords(mws1gt_6,33,178,330).  visible(mws1gt_6,330). orientation(mws1gt_7,167,330). coords(mws1gt_7,39,197,330).  visible(mws1gt_7,330).
orientation(mws1gt_6,146,331). coords(mws1gt_6,33,179,331).  visible(mws1gt_6,331). orientation(mws1gt_7,167,331). coords(mws1gt_7,40,197,331).  visible(mws1gt_7,331).
orientation(mws1gt_6,146,332). coords(mws1gt_6,33,179,332).  visible(mws1gt_6,332). orientation(mws1gt_7,167,332). coords(mws1gt_7,40,196,332).  visible(mws1gt_7,332).
orientation(mws1gt_6,146,333). coords(mws1gt_6,33,179,333).  visible(mws1gt_6,333). orientation(mws1gt_7,167,333). coords(mws1gt_7,41,196,333).  visible(mws1gt_7,333).
orientation(mws1gt_6,146,334). coords(mws1gt_6,33,180,334).  visible(mws1gt_6,334). orientation(mws1gt_7,167,334). coords(mws1gt_7,41,196,334).  visible(mws1gt_7,334).
orientation(mws1gt_6,146,335). coords(mws1gt_6,32,182,335).  visible(mws1gt_6,335). orientation(mws1gt_7,167,335). coords(mws1gt_7,41,195,335).  visible(mws1gt_7,335).
orientation(mws1gt_6,146,336). coords(mws1gt_6,32,183,336).  visible(mws1gt_6,336). orientation(mws1gt_7,167,336). coords(mws1gt_7,41,195,336).  visible(mws1gt_7,336).
orientation(mws1gt_6,146,337). coords(mws1gt_6,32,184,337).  visible(mws1gt_6,337). orientation(mws1gt_7,167,337). coords(mws1gt_7,41,194,337).  visible(mws1gt_7,337).
orientation(mws1gt_6,146,338). coords(mws1gt_6,32,185,338).  visible(mws1gt_6,338). orientation(mws1gt_7,167,338). coords(mws1gt_7,41,193,338).  visible(mws1gt_7,338).
orientation(mws1gt_6,146,339). coords(mws1gt_6,32,186,339).  visible(mws1gt_6,339). orientation(mws1gt_7,167,339). coords(mws1gt_7,41,193,339).  visible(mws1gt_7,339).
orientation(mws1gt_6,160,340). coords(mws1gt_6,32,187,340).  visible(mws1gt_6,340). orientation(mws1gt_7,167,340). coords(mws1gt_7,41,193,340).  visible(mws1gt_7,340).
orientation(mws1gt_6,160,341). coords(mws1gt_6,32,187,341).  visible(mws1gt_6,341). orientation(mws1gt_7,167,341). coords(mws1gt_7,41,192,341).  visible(mws1gt_7,341).
orientation(mws1gt_6,160,342). coords(mws1gt_6,32,188,342).  visible(mws1gt_6,342). orientation(mws1gt_7,174,342). coords(mws1gt_7,41,191,342).  visible(mws1gt_7,342).
orientation(mws1gt_6,160,343). coords(mws1gt_6,32,189,343).  visible(mws1gt_6,343). orientation(mws1gt_7,174,343). coords(mws1gt_7,42,191,343).  visible(mws1gt_7,343).
orientation(mws1gt_6,160,344). coords(mws1gt_6,32,189,344).  visible(mws1gt_6,344). orientation(mws1gt_7,174,344). coords(mws1gt_7,44,190,344).  visible(mws1gt_7,344).
orientation(mws1gt_6,160,345). coords(mws1gt_6,32,189,345).  visible(mws1gt_6,345). orientation(mws1gt_7,174,345). coords(mws1gt_7,45,190,345).  visible(mws1gt_7,345).
orientation(mws1gt_6,160,346). coords(mws1gt_6,32,190,346).  visible(mws1gt_6,346). orientation(mws1gt_7,174,346). coords(mws1gt_7,45,190,346).  visible(mws1gt_7,346).
orientation(mws1gt_6,160,347). coords(mws1gt_6,32,191,347).  visible(mws1gt_6,347). orientation(mws1gt_7,174,347). coords(mws1gt_7,44,190,347).  visible(mws1gt_7,347).
orientation(mws1gt_6,160,348). coords(mws1gt_6,32,192,348).  visible(mws1gt_6,348). orientation(mws1gt_7,174,348). coords(mws1gt_7,45,190,348).  visible(mws1gt_7,348).
orientation(mws1gt_6,160,349). coords(mws1gt_6,32,193,349).  visible(mws1gt_6,349). orientation(mws1gt_7,174,349). coords(mws1gt_7,46,189,349).  visible(mws1gt_7,349).
orientation(mws1gt_6,160,350). coords(mws1gt_6,32,194,350).  visible(mws1gt_6,350). orientation(mws1gt_7,174,350). coords(mws1gt_7,46,188,350).  visible(mws1gt_7,350).
orientation(mws1gt_6,160,351). coords(mws1gt_6,32,196,351).  visible(mws1gt_6,351). orientation(mws1gt_7,174,351). coords(mws1gt_7,46,188,351).  visible(mws1gt_7,351).
orientation(mws1gt_6,160,352). coords(mws1gt_6,32,196,352).  visible(mws1gt_6,352). orientation(mws1gt_7,174,352). coords(mws1gt_7,46,188,352).  visible(mws1gt_7,352).
happensAt(disappear(mws1gt_6),351). happensAt(disappear(mws1gt_7),351).
1{simpleEvent(T,mws1gt_6,active); simpleEvent(T,mws1gt_6,inactive); simpleEvent(T,mws1gt_6,walking); simpleEvent(T,mws1gt_6,running)}1 :- T=327..351.
1{simpleEvent(T,mws1gt_7,active); simpleEvent(T,mws1gt_7,inactive); simpleEvent(T,mws1gt_7,walking); simpleEvent(T,mws1gt_7,running)}1 :- T=327..351.
:- holdsAt(moving(mws1gt_6,mws1gt_7),327). :- holdsAt(moving(mws1gt_7,mws1gt_6),327).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),328). :- not holdsAt(moving(mws1gt_7,mws1gt_6),328).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),329). :- not holdsAt(moving(mws1gt_7,mws1gt_6),329).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),330). :- not holdsAt(moving(mws1gt_7,mws1gt_6),330).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),331). :- not holdsAt(moving(mws1gt_7,mws1gt_6),331).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),332). :- not holdsAt(moving(mws1gt_7,mws1gt_6),332).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),333). :- not holdsAt(moving(mws1gt_7,mws1gt_6),333).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),334). :- not holdsAt(moving(mws1gt_7,mws1gt_6),334).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),335). :- not holdsAt(moving(mws1gt_7,mws1gt_6),335).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),336). :- not holdsAt(moving(mws1gt_7,mws1gt_6),336).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),337). :- not holdsAt(moving(mws1gt_7,mws1gt_6),337).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),338). :- not holdsAt(moving(mws1gt_7,mws1gt_6),338).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),339). :- not holdsAt(moving(mws1gt_7,mws1gt_6),339).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),340). :- not holdsAt(moving(mws1gt_7,mws1gt_6),340).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),341). :- not holdsAt(moving(mws1gt_7,mws1gt_6),341).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),342). :- not holdsAt(moving(mws1gt_7,mws1gt_6),342).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),343). :- not holdsAt(moving(mws1gt_7,mws1gt_6),343).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),344). :- not holdsAt(moving(mws1gt_7,mws1gt_6),344).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),345). :- not holdsAt(moving(mws1gt_7,mws1gt_6),345).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),346). :- not holdsAt(moving(mws1gt_7,mws1gt_6),346).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),347). :- not holdsAt(moving(mws1gt_7,mws1gt_6),347).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),348). :- not holdsAt(moving(mws1gt_7,mws1gt_6),348).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),349). :- not holdsAt(moving(mws1gt_7,mws1gt_6),349).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),350). :- not holdsAt(moving(mws1gt_7,mws1gt_6),350).
:- not holdsAt(moving(mws1gt_6,mws1gt_7),351). :- not holdsAt(moving(mws1gt_7,mws1gt_6),351).
:- holdsAt(moving(mws1gt_6,mws1gt_7),352). :- holdsAt(moving(mws1gt_7,mws1gt_6),352).
:- simpleEvent(T,P,inactive), T>327 , T<351.
:- simpleEvent(T,P,running), T>327 , T<351.

#show simpleEvent/3.