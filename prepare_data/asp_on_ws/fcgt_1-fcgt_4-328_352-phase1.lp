
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fcgt_1,156,328). coords(fcgt_1,103,137,328).  visible(fcgt_1,328). happensAt(active(fcgt_1),328). orientation(fcgt_4,166,328). coords(fcgt_4,82,206,328).  visible(fcgt_4,328). happensAt(inactive(fcgt_4),328).
orientation(fcgt_1,160,329). coords(fcgt_1,103,136,329).  visible(fcgt_1,329). happensAt(active(fcgt_1),329). orientation(fcgt_4,166,329). coords(fcgt_4,82,206,329).  visible(fcgt_4,329). happensAt(inactive(fcgt_4),329).
orientation(fcgt_1,163,330). coords(fcgt_1,103,135,330).  visible(fcgt_1,330). happensAt(active(fcgt_1),330). orientation(fcgt_4,166,330). coords(fcgt_4,82,206,330).  visible(fcgt_4,330). happensAt(inactive(fcgt_4),330).
orientation(fcgt_1,160,331). coords(fcgt_1,101,134,331).  visible(fcgt_1,331). happensAt(active(fcgt_1),331). orientation(fcgt_4,166,331). coords(fcgt_4,82,206,331).  visible(fcgt_4,331). happensAt(inactive(fcgt_4),331).
orientation(fcgt_1,160,332). coords(fcgt_1,99,134,332).  visible(fcgt_1,332). happensAt(active(fcgt_1),332). orientation(fcgt_4,166,332). coords(fcgt_4,82,206,332).  visible(fcgt_4,332). happensAt(inactive(fcgt_4),332).
orientation(fcgt_1,161,333). coords(fcgt_1,99,135,333).  visible(fcgt_1,333). happensAt(active(fcgt_1),333). orientation(fcgt_4,166,333). coords(fcgt_4,82,206,333).  visible(fcgt_4,333). happensAt(inactive(fcgt_4),333).
orientation(fcgt_1,158,334). coords(fcgt_1,98,135,334).  visible(fcgt_1,334). happensAt(active(fcgt_1),334). orientation(fcgt_4,166,334). coords(fcgt_4,82,206,334).  visible(fcgt_4,334). happensAt(inactive(fcgt_4),334).
orientation(fcgt_1,153,335). coords(fcgt_1,97,134,335).  visible(fcgt_1,335). happensAt(active(fcgt_1),335). orientation(fcgt_4,166,335). coords(fcgt_4,82,206,335).  visible(fcgt_4,335). happensAt(inactive(fcgt_4),335).
orientation(fcgt_1,151,336). coords(fcgt_1,97,134,336).  visible(fcgt_1,336). happensAt(active(fcgt_1),336). orientation(fcgt_4,166,336). coords(fcgt_4,82,206,336).  visible(fcgt_4,336). happensAt(inactive(fcgt_4),336).
orientation(fcgt_1,147,337). coords(fcgt_1,96,133,337).  visible(fcgt_1,337). happensAt(active(fcgt_1),337). orientation(fcgt_4,166,337). coords(fcgt_4,82,206,337).  visible(fcgt_4,337). happensAt(inactive(fcgt_4),337).
orientation(fcgt_1,147,338). coords(fcgt_1,96,132,338).  visible(fcgt_1,338). happensAt(active(fcgt_1),338). orientation(fcgt_4,166,338). coords(fcgt_4,82,206,338).  visible(fcgt_4,338). happensAt(inactive(fcgt_4),338).
orientation(fcgt_1,147,339). coords(fcgt_1,96,130,339).  visible(fcgt_1,339). happensAt(active(fcgt_1),339). orientation(fcgt_4,166,339). coords(fcgt_4,82,206,339).  visible(fcgt_4,339). happensAt(inactive(fcgt_4),339).
orientation(fcgt_1,147,340). coords(fcgt_1,96,129,340).  visible(fcgt_1,340). happensAt(active(fcgt_1),340). orientation(fcgt_4,166,340). coords(fcgt_4,82,206,340).  visible(fcgt_4,340). happensAt(inactive(fcgt_4),340).
orientation(fcgt_1,150,341). coords(fcgt_1,96,129,341).  visible(fcgt_1,341). happensAt(active(fcgt_1),341). orientation(fcgt_4,166,341). coords(fcgt_4,82,206,341).  visible(fcgt_4,341). happensAt(inactive(fcgt_4),341).
orientation(fcgt_1,148,342). coords(fcgt_1,96,129,342).  visible(fcgt_1,342). happensAt(active(fcgt_1),342). orientation(fcgt_4,166,342). coords(fcgt_4,82,206,342).  visible(fcgt_4,342). happensAt(inactive(fcgt_4),342).
orientation(fcgt_1,146,343). coords(fcgt_1,96,130,343).  visible(fcgt_1,343). happensAt(active(fcgt_1),343). orientation(fcgt_4,166,343). coords(fcgt_4,82,206,343).  visible(fcgt_4,343). happensAt(inactive(fcgt_4),343).
orientation(fcgt_1,146,344). coords(fcgt_1,96,130,344).  visible(fcgt_1,344). happensAt(active(fcgt_1),344). orientation(fcgt_4,166,344). coords(fcgt_4,82,206,344).  visible(fcgt_4,344). happensAt(inactive(fcgt_4),344).
orientation(fcgt_1,146,345). coords(fcgt_1,97,129,345).  visible(fcgt_1,345). happensAt(active(fcgt_1),345). orientation(fcgt_4,166,345). coords(fcgt_4,82,206,345).  visible(fcgt_4,345). happensAt(inactive(fcgt_4),345).
orientation(fcgt_1,146,346). coords(fcgt_1,97,129,346).  visible(fcgt_1,346). happensAt(active(fcgt_1),346). orientation(fcgt_4,166,346). coords(fcgt_4,82,206,346).  visible(fcgt_4,346). happensAt(inactive(fcgt_4),346).
orientation(fcgt_1,141,347). coords(fcgt_1,97,127,347).  visible(fcgt_1,347). happensAt(active(fcgt_1),347). orientation(fcgt_4,166,347). coords(fcgt_4,82,206,347).  visible(fcgt_4,347). happensAt(inactive(fcgt_4),347).
orientation(fcgt_1,141,348). coords(fcgt_1,97,126,348).  visible(fcgt_1,348). happensAt(active(fcgt_1),348). orientation(fcgt_4,166,348). coords(fcgt_4,82,206,348).  visible(fcgt_4,348). happensAt(inactive(fcgt_4),348).
orientation(fcgt_1,141,349). coords(fcgt_1,97,124,349).  visible(fcgt_1,349). happensAt(active(fcgt_1),349). orientation(fcgt_4,166,349). coords(fcgt_4,82,206,349).  visible(fcgt_4,349). happensAt(inactive(fcgt_4),349).
orientation(fcgt_1,141,350). coords(fcgt_1,97,123,350).  visible(fcgt_1,350). happensAt(active(fcgt_1),350). orientation(fcgt_4,166,350). coords(fcgt_4,82,206,350).  visible(fcgt_4,350). happensAt(inactive(fcgt_4),350).
orientation(fcgt_1,141,351). coords(fcgt_1,96,122,351).  visible(fcgt_1,351). happensAt(active(fcgt_1),351). orientation(fcgt_4,166,351). coords(fcgt_4,82,206,351).  visible(fcgt_4,351). happensAt(inactive(fcgt_4),351).
orientation(fcgt_1,141,352). coords(fcgt_1,96,123,352).  visible(fcgt_1,352). happensAt(active(fcgt_1),352). orientation(fcgt_4,166,352). coords(fcgt_4,82,206,352).  visible(fcgt_4,352). happensAt(inactive(fcgt_4),352).
#show holdsAt/2.