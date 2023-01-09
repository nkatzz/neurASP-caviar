
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

	orientation(mws1gt_3,165,327). coords(mws1gt_3,62,143,327).  visible(mws1gt_3,327). happensAt(walking(mws1gt_3),327). orientation(mws1gt_7,162,327). coords(mws1gt_7,38,199,327).  visible(mws1gt_7,327). happensAt(walking(mws1gt_7),327).
orientation(mws1gt_3,165,328). coords(mws1gt_3,61,143,328).  visible(mws1gt_3,328). happensAt(walking(mws1gt_3),328). orientation(mws1gt_7,167,328). coords(mws1gt_7,38,197,328).  visible(mws1gt_7,328). happensAt(walking(mws1gt_7),328).
orientation(mws1gt_3,165,329). coords(mws1gt_3,61,143,329).  visible(mws1gt_3,329). happensAt(walking(mws1gt_3),329). orientation(mws1gt_7,167,329). coords(mws1gt_7,39,197,329).  visible(mws1gt_7,329). happensAt(walking(mws1gt_7),329).
orientation(mws1gt_3,165,330). coords(mws1gt_3,60,144,330).  visible(mws1gt_3,330). happensAt(walking(mws1gt_3),330). orientation(mws1gt_7,167,330). coords(mws1gt_7,39,197,330).  visible(mws1gt_7,330). happensAt(walking(mws1gt_7),330).
orientation(mws1gt_3,165,331). coords(mws1gt_3,60,146,331).  visible(mws1gt_3,331). happensAt(walking(mws1gt_3),331). orientation(mws1gt_7,167,331). coords(mws1gt_7,40,197,331).  visible(mws1gt_7,331). happensAt(walking(mws1gt_7),331).
orientation(mws1gt_3,165,332). coords(mws1gt_3,60,146,332).  visible(mws1gt_3,332). happensAt(walking(mws1gt_3),332). orientation(mws1gt_7,167,332). coords(mws1gt_7,40,196,332).  visible(mws1gt_7,332). happensAt(walking(mws1gt_7),332).
orientation(mws1gt_3,165,333). coords(mws1gt_3,60,147,333).  visible(mws1gt_3,333). happensAt(walking(mws1gt_3),333). orientation(mws1gt_7,167,333). coords(mws1gt_7,41,196,333).  visible(mws1gt_7,333). happensAt(walking(mws1gt_7),333).
orientation(mws1gt_3,165,334). coords(mws1gt_3,60,147,334).  visible(mws1gt_3,334). happensAt(walking(mws1gt_3),334). orientation(mws1gt_7,167,334). coords(mws1gt_7,41,196,334).  visible(mws1gt_7,334). happensAt(walking(mws1gt_7),334).
orientation(mws1gt_3,165,335). coords(mws1gt_3,60,148,335).  visible(mws1gt_3,335). happensAt(walking(mws1gt_3),335). orientation(mws1gt_7,167,335). coords(mws1gt_7,41,195,335).  visible(mws1gt_7,335). happensAt(walking(mws1gt_7),335).
orientation(mws1gt_3,165,336). coords(mws1gt_3,60,149,336).  visible(mws1gt_3,336). happensAt(walking(mws1gt_3),336). orientation(mws1gt_7,167,336). coords(mws1gt_7,41,195,336).  visible(mws1gt_7,336). happensAt(walking(mws1gt_7),336).
orientation(mws1gt_3,165,337). coords(mws1gt_3,59,149,337).  visible(mws1gt_3,337). happensAt(walking(mws1gt_3),337). orientation(mws1gt_7,167,337). coords(mws1gt_7,41,194,337).  visible(mws1gt_7,337). happensAt(walking(mws1gt_7),337).
orientation(mws1gt_3,165,338). coords(mws1gt_3,59,149,338).  visible(mws1gt_3,338). happensAt(walking(mws1gt_3),338). orientation(mws1gt_7,167,338). coords(mws1gt_7,41,193,338).  visible(mws1gt_7,338). happensAt(walking(mws1gt_7),338).
orientation(mws1gt_3,165,339). coords(mws1gt_3,59,149,339).  visible(mws1gt_3,339). happensAt(walking(mws1gt_3),339). orientation(mws1gt_7,167,339). coords(mws1gt_7,41,193,339).  visible(mws1gt_7,339). happensAt(walking(mws1gt_7),339).
orientation(mws1gt_3,165,340). coords(mws1gt_3,58,150,340).  visible(mws1gt_3,340). happensAt(walking(mws1gt_3),340). orientation(mws1gt_7,167,340). coords(mws1gt_7,41,193,340).  visible(mws1gt_7,340). happensAt(walking(mws1gt_7),340).
orientation(mws1gt_3,165,341). coords(mws1gt_3,57,150,341).  visible(mws1gt_3,341). happensAt(walking(mws1gt_3),341). orientation(mws1gt_7,167,341). coords(mws1gt_7,41,192,341).  visible(mws1gt_7,341). happensAt(walking(mws1gt_7),341).
orientation(mws1gt_3,165,342). coords(mws1gt_3,56,150,342).  visible(mws1gt_3,342). happensAt(walking(mws1gt_3),342). orientation(mws1gt_7,174,342). coords(mws1gt_7,41,191,342).  visible(mws1gt_7,342). happensAt(walking(mws1gt_7),342).
orientation(mws1gt_3,165,343). coords(mws1gt_3,55,151,343).  visible(mws1gt_3,343). happensAt(walking(mws1gt_3),343). orientation(mws1gt_7,174,343). coords(mws1gt_7,42,191,343).  visible(mws1gt_7,343). happensAt(walking(mws1gt_7),343).
orientation(mws1gt_3,165,344). coords(mws1gt_3,55,151,344).  visible(mws1gt_3,344). happensAt(walking(mws1gt_3),344). orientation(mws1gt_7,174,344). coords(mws1gt_7,44,190,344).  visible(mws1gt_7,344). happensAt(walking(mws1gt_7),344).
orientation(mws1gt_3,165,345). coords(mws1gt_3,55,151,345).  visible(mws1gt_3,345). happensAt(walking(mws1gt_3),345). orientation(mws1gt_7,174,345). coords(mws1gt_7,45,190,345).  visible(mws1gt_7,345). happensAt(walking(mws1gt_7),345).
orientation(mws1gt_3,165,346). coords(mws1gt_3,54,152,346).  visible(mws1gt_3,346). happensAt(walking(mws1gt_3),346). orientation(mws1gt_7,174,346). coords(mws1gt_7,45,190,346).  visible(mws1gt_7,346). happensAt(walking(mws1gt_7),346).
orientation(mws1gt_3,165,347). coords(mws1gt_3,54,152,347).  visible(mws1gt_3,347). happensAt(walking(mws1gt_3),347). orientation(mws1gt_7,174,347). coords(mws1gt_7,44,190,347).  visible(mws1gt_7,347). happensAt(walking(mws1gt_7),347).
orientation(mws1gt_3,165,348). coords(mws1gt_3,54,152,348).  visible(mws1gt_3,348). happensAt(walking(mws1gt_3),348). orientation(mws1gt_7,174,348). coords(mws1gt_7,45,190,348).  visible(mws1gt_7,348). happensAt(walking(mws1gt_7),348).
orientation(mws1gt_3,165,349). coords(mws1gt_3,54,153,349).  visible(mws1gt_3,349). happensAt(walking(mws1gt_3),349). orientation(mws1gt_7,174,349). coords(mws1gt_7,46,189,349).  visible(mws1gt_7,349). happensAt(walking(mws1gt_7),349).
orientation(mws1gt_3,165,350). coords(mws1gt_3,54,154,350).  visible(mws1gt_3,350). happensAt(walking(mws1gt_3),350). orientation(mws1gt_7,174,350). coords(mws1gt_7,46,188,350).  visible(mws1gt_7,350). happensAt(walking(mws1gt_7),350).
orientation(mws1gt_3,165,351). coords(mws1gt_3,53,154,351).  visible(mws1gt_3,351). happensAt(walking(mws1gt_3),351). orientation(mws1gt_7,174,351). coords(mws1gt_7,46,188,351).  visible(mws1gt_7,351). happensAt(walking(mws1gt_7),351).
#show holdsAt/2.