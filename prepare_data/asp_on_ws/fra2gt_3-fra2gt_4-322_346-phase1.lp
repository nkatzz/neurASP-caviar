
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

	orientation(fra2gt_3,150,322). coords(fra2gt_3,94,116,322).  visible(fra2gt_3,322). happensAt(walking(fra2gt_3),322). orientation(fra2gt_4,155,322). coords(fra2gt_4,101,121,322).  visible(fra2gt_4,322). happensAt(walking(fra2gt_4),322).
orientation(fra2gt_3,143,323). coords(fra2gt_3,92,116,323).  visible(fra2gt_3,323). happensAt(walking(fra2gt_3),323). orientation(fra2gt_4,155,323). coords(fra2gt_4,101,121,323).  visible(fra2gt_4,323). happensAt(walking(fra2gt_4),323).
orientation(fra2gt_3,143,324). coords(fra2gt_3,91,110,324).  visible(fra2gt_3,324). happensAt(walking(fra2gt_3),324). orientation(fra2gt_4,155,324). coords(fra2gt_4,101,122,324).  visible(fra2gt_4,324). happensAt(walking(fra2gt_4),324).
orientation(fra2gt_3,143,325). coords(fra2gt_3,91,110,325).  visible(fra2gt_3,325). happensAt(walking(fra2gt_3),325). orientation(fra2gt_4,155,325). coords(fra2gt_4,101,122,325).  visible(fra2gt_4,325). happensAt(walking(fra2gt_4),325).
orientation(fra2gt_3,143,326). coords(fra2gt_3,91,110,326).  visible(fra2gt_3,326). happensAt(walking(fra2gt_3),326). orientation(fra2gt_4,155,326). coords(fra2gt_4,101,122,326).  visible(fra2gt_4,326). happensAt(walking(fra2gt_4),326).
orientation(fra2gt_3,143,327). coords(fra2gt_3,90,110,327).  visible(fra2gt_3,327). happensAt(walking(fra2gt_3),327). orientation(fra2gt_4,155,327). coords(fra2gt_4,101,122,327).  visible(fra2gt_4,327). happensAt(walking(fra2gt_4),327).
orientation(fra2gt_3,143,328). coords(fra2gt_3,90,114,328).  visible(fra2gt_3,328). happensAt(walking(fra2gt_3),328). orientation(fra2gt_4,155,328). coords(fra2gt_4,101,123,328).  visible(fra2gt_4,328). happensAt(walking(fra2gt_4),328).
orientation(fra2gt_3,143,329). coords(fra2gt_3,89,114,329).  visible(fra2gt_3,329). happensAt(walking(fra2gt_3),329). orientation(fra2gt_4,155,329). coords(fra2gt_4,101,122,329).  visible(fra2gt_4,329). happensAt(walking(fra2gt_4),329).
orientation(fra2gt_3,143,330). coords(fra2gt_3,88,114,330).  visible(fra2gt_3,330). happensAt(walking(fra2gt_3),330). orientation(fra2gt_4,155,330). coords(fra2gt_4,102,120,330).  visible(fra2gt_4,330). happensAt(walking(fra2gt_4),330).
orientation(fra2gt_3,143,331). coords(fra2gt_3,88,114,331).  visible(fra2gt_3,331). happensAt(walking(fra2gt_3),331). orientation(fra2gt_4,155,331). coords(fra2gt_4,102,119,331).  visible(fra2gt_4,331). happensAt(walking(fra2gt_4),331).
orientation(fra2gt_3,138,332). coords(fra2gt_3,89,114,332).  visible(fra2gt_3,332). happensAt(walking(fra2gt_3),332). orientation(fra2gt_4,155,332). coords(fra2gt_4,101,117,332).  visible(fra2gt_4,332). happensAt(walking(fra2gt_4),332).
orientation(fra2gt_3,138,333). coords(fra2gt_3,89,114,333).  visible(fra2gt_3,333). happensAt(walking(fra2gt_3),333). orientation(fra2gt_4,155,333). coords(fra2gt_4,100,116,333).  visible(fra2gt_4,333). happensAt(walking(fra2gt_4),333).
orientation(fra2gt_3,138,334). coords(fra2gt_3,89,114,334).  visible(fra2gt_3,334). happensAt(walking(fra2gt_3),334). orientation(fra2gt_4,155,334). coords(fra2gt_4,99,116,334).  visible(fra2gt_4,334). happensAt(walking(fra2gt_4),334).
orientation(fra2gt_3,138,335). coords(fra2gt_3,89,113,335).  visible(fra2gt_3,335). happensAt(walking(fra2gt_3),335). orientation(fra2gt_4,155,335). coords(fra2gt_4,99,116,335).  visible(fra2gt_4,335). happensAt(walking(fra2gt_4),335).
orientation(fra2gt_3,138,336). coords(fra2gt_3,89,109,336).  visible(fra2gt_3,336). happensAt(walking(fra2gt_3),336). orientation(fra2gt_4,155,336). coords(fra2gt_4,98,114,336).  visible(fra2gt_4,336). happensAt(walking(fra2gt_4),336).
orientation(fra2gt_3,138,337). coords(fra2gt_3,89,108,337).  visible(fra2gt_3,337). happensAt(walking(fra2gt_3),337). orientation(fra2gt_4,155,337). coords(fra2gt_4,97,112,337).  visible(fra2gt_4,337). happensAt(walking(fra2gt_4),337).
orientation(fra2gt_3,138,338). coords(fra2gt_3,89,108,338).  visible(fra2gt_3,338). happensAt(walking(fra2gt_3),338). orientation(fra2gt_4,155,338). coords(fra2gt_4,98,110,338).  visible(fra2gt_4,338). happensAt(walking(fra2gt_4),338).
orientation(fra2gt_3,138,339). coords(fra2gt_3,91,109,339).  visible(fra2gt_3,339). happensAt(walking(fra2gt_3),339). orientation(fra2gt_4,155,339). coords(fra2gt_4,98,110,339).  visible(fra2gt_4,339). happensAt(walking(fra2gt_4),339).
orientation(fra2gt_3,138,340). coords(fra2gt_3,91,109,340).  visible(fra2gt_3,340). happensAt(walking(fra2gt_3),340). orientation(fra2gt_4,155,340). coords(fra2gt_4,98,111,340).  visible(fra2gt_4,340). happensAt(walking(fra2gt_4),340).
orientation(fra2gt_3,138,341). coords(fra2gt_3,91,109,341).  visible(fra2gt_3,341). happensAt(walking(fra2gt_3),341). orientation(fra2gt_4,155,341). coords(fra2gt_4,97,113,341).  visible(fra2gt_4,341). happensAt(walking(fra2gt_4),341).
orientation(fra2gt_3,138,342). coords(fra2gt_3,91,110,342).  visible(fra2gt_3,342). happensAt(walking(fra2gt_3),342). orientation(fra2gt_4,155,342). coords(fra2gt_4,97,115,342).  visible(fra2gt_4,342). happensAt(walking(fra2gt_4),342).
orientation(fra2gt_3,138,343). coords(fra2gt_3,91,110,343).  visible(fra2gt_3,343). happensAt(walking(fra2gt_3),343). orientation(fra2gt_4,155,343). coords(fra2gt_4,97,117,343).  visible(fra2gt_4,343). happensAt(walking(fra2gt_4),343).
orientation(fra2gt_3,138,344). coords(fra2gt_3,91,110,344).  visible(fra2gt_3,344). happensAt(walking(fra2gt_3),344). orientation(fra2gt_4,155,344). coords(fra2gt_4,97,118,344).  visible(fra2gt_4,344). happensAt(walking(fra2gt_4),344).
orientation(fra2gt_3,138,345). coords(fra2gt_3,91,111,345).  visible(fra2gt_3,345). happensAt(walking(fra2gt_3),345). orientation(fra2gt_4,155,345). coords(fra2gt_4,97,119,345).  visible(fra2gt_4,345). happensAt(walking(fra2gt_4),345).
orientation(fra2gt_3,138,346). coords(fra2gt_3,90,111,346).  visible(fra2gt_3,346). happensAt(walking(fra2gt_3),346). orientation(fra2gt_4,155,346). coords(fra2gt_4,98,119,346).  visible(fra2gt_4,346). happensAt(walking(fra2gt_4),346).
#show holdsAt/2.