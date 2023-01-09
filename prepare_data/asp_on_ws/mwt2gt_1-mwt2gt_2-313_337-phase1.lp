
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

	orientation(mwt2gt_1,143,313). coords(mwt2gt_1,138,75,313).  visible(mwt2gt_1,313). happensAt(inactive(mwt2gt_1),313). orientation(mwt2gt_2,146,313). coords(mwt2gt_2,144,89,313).  visible(mwt2gt_2,313). happensAt(active(mwt2gt_2),313).
orientation(mwt2gt_1,143,314). coords(mwt2gt_1,138,75,314).  visible(mwt2gt_1,314). happensAt(inactive(mwt2gt_1),314). orientation(mwt2gt_2,146,314). coords(mwt2gt_2,144,89,314).  visible(mwt2gt_2,314). happensAt(active(mwt2gt_2),314).
orientation(mwt2gt_1,143,315). coords(mwt2gt_1,138,75,315).  visible(mwt2gt_1,315). happensAt(inactive(mwt2gt_1),315). orientation(mwt2gt_2,146,315). coords(mwt2gt_2,143,90,315).  visible(mwt2gt_2,315). happensAt(active(mwt2gt_2),315).
orientation(mwt2gt_1,143,316). coords(mwt2gt_1,138,75,316).  visible(mwt2gt_1,316). happensAt(inactive(mwt2gt_1),316). orientation(mwt2gt_2,146,316). coords(mwt2gt_2,143,89,316).  visible(mwt2gt_2,316). happensAt(active(mwt2gt_2),316).
orientation(mwt2gt_1,143,317). coords(mwt2gt_1,138,75,317).  visible(mwt2gt_1,317). happensAt(inactive(mwt2gt_1),317). orientation(mwt2gt_2,146,317). coords(mwt2gt_2,142,89,317).  visible(mwt2gt_2,317). happensAt(active(mwt2gt_2),317).
orientation(mwt2gt_1,143,318). coords(mwt2gt_1,138,75,318).  visible(mwt2gt_1,318). happensAt(inactive(mwt2gt_1),318). orientation(mwt2gt_2,146,318). coords(mwt2gt_2,142,89,318).  visible(mwt2gt_2,318). happensAt(active(mwt2gt_2),318).
orientation(mwt2gt_1,143,319). coords(mwt2gt_1,138,75,319).  visible(mwt2gt_1,319). happensAt(inactive(mwt2gt_1),319). orientation(mwt2gt_2,146,319). coords(mwt2gt_2,142,89,319).  visible(mwt2gt_2,319). happensAt(active(mwt2gt_2),319).
orientation(mwt2gt_1,143,320). coords(mwt2gt_1,138,75,320).  visible(mwt2gt_1,320). happensAt(inactive(mwt2gt_1),320). orientation(mwt2gt_2,146,320). coords(mwt2gt_2,141,89,320).  visible(mwt2gt_2,320). happensAt(active(mwt2gt_2),320).
orientation(mwt2gt_1,143,321). coords(mwt2gt_1,138,75,321).  visible(mwt2gt_1,321). happensAt(inactive(mwt2gt_1),321). orientation(mwt2gt_2,146,321). coords(mwt2gt_2,141,88,321).  visible(mwt2gt_2,321). happensAt(active(mwt2gt_2),321).
orientation(mwt2gt_1,143,322). coords(mwt2gt_1,138,75,322).  visible(mwt2gt_1,322). happensAt(inactive(mwt2gt_1),322). orientation(mwt2gt_2,146,322). coords(mwt2gt_2,142,88,322).  visible(mwt2gt_2,322). happensAt(active(mwt2gt_2),322).
orientation(mwt2gt_1,143,323). coords(mwt2gt_1,138,75,323).  visible(mwt2gt_1,323). happensAt(inactive(mwt2gt_1),323). orientation(mwt2gt_2,146,323). coords(mwt2gt_2,142,88,323).  visible(mwt2gt_2,323). happensAt(active(mwt2gt_2),323).
orientation(mwt2gt_1,143,324). coords(mwt2gt_1,138,75,324).  visible(mwt2gt_1,324). happensAt(inactive(mwt2gt_1),324). orientation(mwt2gt_2,146,324). coords(mwt2gt_2,142,87,324).  visible(mwt2gt_2,324). happensAt(active(mwt2gt_2),324).
orientation(mwt2gt_1,143,325). coords(mwt2gt_1,138,75,325).  visible(mwt2gt_1,325). happensAt(active(mwt2gt_1),325). orientation(mwt2gt_2,145,325). coords(mwt2gt_2,142,87,325).  visible(mwt2gt_2,325). happensAt(active(mwt2gt_2),325).
orientation(mwt2gt_1,143,326). coords(mwt2gt_1,138,75,326).  visible(mwt2gt_1,326). happensAt(active(mwt2gt_1),326). orientation(mwt2gt_2,144,326). coords(mwt2gt_2,142,87,326).  visible(mwt2gt_2,326). happensAt(active(mwt2gt_2),326).
orientation(mwt2gt_1,143,327). coords(mwt2gt_1,138,75,327).  visible(mwt2gt_1,327). happensAt(active(mwt2gt_1),327). orientation(mwt2gt_2,143,327). coords(mwt2gt_2,143,87,327).  visible(mwt2gt_2,327). happensAt(active(mwt2gt_2),327).
orientation(mwt2gt_1,143,328). coords(mwt2gt_1,138,76,328).  visible(mwt2gt_1,328). happensAt(active(mwt2gt_1),328). orientation(mwt2gt_2,145,328). coords(mwt2gt_2,143,87,328).  visible(mwt2gt_2,328). happensAt(active(mwt2gt_2),328).
orientation(mwt2gt_1,143,329). coords(mwt2gt_1,138,76,329).  visible(mwt2gt_1,329). happensAt(active(mwt2gt_1),329). orientation(mwt2gt_2,145,329). coords(mwt2gt_2,143,87,329).  visible(mwt2gt_2,329). happensAt(active(mwt2gt_2),329).
orientation(mwt2gt_1,143,330). coords(mwt2gt_1,138,76,330).  visible(mwt2gt_1,330). happensAt(active(mwt2gt_1),330). orientation(mwt2gt_2,145,330). coords(mwt2gt_2,143,87,330).  visible(mwt2gt_2,330). happensAt(active(mwt2gt_2),330).
orientation(mwt2gt_1,143,331). coords(mwt2gt_1,138,76,331).  visible(mwt2gt_1,331). happensAt(active(mwt2gt_1),331). orientation(mwt2gt_2,145,331). coords(mwt2gt_2,143,87,331).  visible(mwt2gt_2,331). happensAt(active(mwt2gt_2),331).
orientation(mwt2gt_1,143,332). coords(mwt2gt_1,138,76,332).  visible(mwt2gt_1,332). happensAt(active(mwt2gt_1),332). orientation(mwt2gt_2,145,332). coords(mwt2gt_2,143,87,332).  visible(mwt2gt_2,332). happensAt(active(mwt2gt_2),332).
orientation(mwt2gt_1,143,333). coords(mwt2gt_1,137,76,333).  visible(mwt2gt_1,333). happensAt(active(mwt2gt_1),333). orientation(mwt2gt_2,145,333). coords(mwt2gt_2,143,87,333).  visible(mwt2gt_2,333). happensAt(active(mwt2gt_2),333).
orientation(mwt2gt_1,143,334). coords(mwt2gt_1,137,76,334).  visible(mwt2gt_1,334). happensAt(active(mwt2gt_1),334). orientation(mwt2gt_2,145,334). coords(mwt2gt_2,142,87,334).  visible(mwt2gt_2,334). happensAt(active(mwt2gt_2),334).
orientation(mwt2gt_1,143,335). coords(mwt2gt_1,137,76,335).  visible(mwt2gt_1,335). happensAt(active(mwt2gt_1),335). orientation(mwt2gt_2,145,335). coords(mwt2gt_2,142,87,335).  visible(mwt2gt_2,335). happensAt(active(mwt2gt_2),335).
orientation(mwt2gt_1,143,336). coords(mwt2gt_1,137,76,336).  visible(mwt2gt_1,336). happensAt(active(mwt2gt_1),336). orientation(mwt2gt_2,145,336). coords(mwt2gt_2,142,87,336).  visible(mwt2gt_2,336). happensAt(active(mwt2gt_2),336).
orientation(mwt2gt_1,143,337). coords(mwt2gt_1,137,76,337).  visible(mwt2gt_1,337). happensAt(active(mwt2gt_1),337). orientation(mwt2gt_2,145,337). coords(mwt2gt_2,142,87,337).  visible(mwt2gt_2,337). happensAt(active(mwt2gt_2),337).
#show holdsAt/2.