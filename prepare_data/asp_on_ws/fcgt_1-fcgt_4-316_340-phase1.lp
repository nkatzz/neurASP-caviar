
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

	orientation(fcgt_1,153,316). coords(fcgt_1,98,131,316).  visible(fcgt_1,316). happensAt(active(fcgt_1),316). orientation(fcgt_4,166,316). coords(fcgt_4,82,206,316).  visible(fcgt_4,316). happensAt(inactive(fcgt_4),316).
orientation(fcgt_1,153,317). coords(fcgt_1,99,131,317).  visible(fcgt_1,317). happensAt(active(fcgt_1),317). orientation(fcgt_4,166,317). coords(fcgt_4,82,206,317).  visible(fcgt_4,317). happensAt(inactive(fcgt_4),317).
orientation(fcgt_1,153,318). coords(fcgt_1,100,130,318).  visible(fcgt_1,318). happensAt(active(fcgt_1),318). orientation(fcgt_4,166,318). coords(fcgt_4,82,206,318).  visible(fcgt_4,318). happensAt(inactive(fcgt_4),318).
orientation(fcgt_1,153,319). coords(fcgt_1,101,130,319).  visible(fcgt_1,319). happensAt(active(fcgt_1),319). orientation(fcgt_4,166,319). coords(fcgt_4,82,206,319).  visible(fcgt_4,319). happensAt(inactive(fcgt_4),319).
orientation(fcgt_1,153,320). coords(fcgt_1,102,130,320).  visible(fcgt_1,320). happensAt(active(fcgt_1),320). orientation(fcgt_4,166,320). coords(fcgt_4,82,206,320).  visible(fcgt_4,320). happensAt(inactive(fcgt_4),320).
orientation(fcgt_1,153,321). coords(fcgt_1,103,131,321).  visible(fcgt_1,321). happensAt(active(fcgt_1),321). orientation(fcgt_4,166,321). coords(fcgt_4,82,206,321).  visible(fcgt_4,321). happensAt(inactive(fcgt_4),321).
orientation(fcgt_1,153,322). coords(fcgt_1,104,133,322).  visible(fcgt_1,322). happensAt(active(fcgt_1),322). orientation(fcgt_4,166,322). coords(fcgt_4,82,206,322).  visible(fcgt_4,322). happensAt(inactive(fcgt_4),322).
orientation(fcgt_1,153,323). coords(fcgt_1,103,134,323).  visible(fcgt_1,323). happensAt(active(fcgt_1),323). orientation(fcgt_4,166,323). coords(fcgt_4,82,206,323).  visible(fcgt_4,323). happensAt(inactive(fcgt_4),323).
orientation(fcgt_1,153,324). coords(fcgt_1,103,135,324).  visible(fcgt_1,324). happensAt(active(fcgt_1),324). orientation(fcgt_4,166,324). coords(fcgt_4,82,206,324).  visible(fcgt_4,324). happensAt(inactive(fcgt_4),324).
orientation(fcgt_1,153,325). coords(fcgt_1,103,136,325).  visible(fcgt_1,325). happensAt(active(fcgt_1),325). orientation(fcgt_4,166,325). coords(fcgt_4,82,206,325).  visible(fcgt_4,325). happensAt(inactive(fcgt_4),325).
orientation(fcgt_1,155,326). coords(fcgt_1,103,136,326).  visible(fcgt_1,326). happensAt(active(fcgt_1),326). orientation(fcgt_4,166,326). coords(fcgt_4,82,206,326).  visible(fcgt_4,326). happensAt(inactive(fcgt_4),326).
orientation(fcgt_1,154,327). coords(fcgt_1,103,137,327).  visible(fcgt_1,327). happensAt(active(fcgt_1),327). orientation(fcgt_4,166,327). coords(fcgt_4,82,206,327).  visible(fcgt_4,327). happensAt(inactive(fcgt_4),327).
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
#show holdsAt/2.