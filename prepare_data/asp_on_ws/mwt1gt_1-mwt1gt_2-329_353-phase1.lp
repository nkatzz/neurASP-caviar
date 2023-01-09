
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

	orientation(mwt1gt_1,160,329). coords(mwt1gt_1,118,151,329).  visible(mwt1gt_1,329). happensAt(walking(mwt1gt_1),329). orientation(mwt1gt_2,140,329). coords(mwt1gt_2,129,126,329).  visible(mwt1gt_2,329). happensAt(walking(mwt1gt_2),329).
orientation(mwt1gt_1,160,330). coords(mwt1gt_1,119,150,330).  visible(mwt1gt_1,330). happensAt(active(mwt1gt_1),330). orientation(mwt1gt_2,140,330). coords(mwt1gt_2,129,126,330).  visible(mwt1gt_2,330). happensAt(walking(mwt1gt_2),330).
orientation(mwt1gt_1,160,331). coords(mwt1gt_1,119,150,331).  visible(mwt1gt_1,331). happensAt(active(mwt1gt_1),331). orientation(mwt1gt_2,140,331). coords(mwt1gt_2,129,126,331).  visible(mwt1gt_2,331). happensAt(walking(mwt1gt_2),331).
orientation(mwt1gt_1,160,332). coords(mwt1gt_1,119,150,332).  visible(mwt1gt_1,332). happensAt(active(mwt1gt_1),332). orientation(mwt1gt_2,140,332). coords(mwt1gt_2,128,126,332).  visible(mwt1gt_2,332). happensAt(active(mwt1gt_2),332).
orientation(mwt1gt_1,160,333). coords(mwt1gt_1,119,150,333).  visible(mwt1gt_1,333). happensAt(active(mwt1gt_1),333). orientation(mwt1gt_2,140,333). coords(mwt1gt_2,128,127,333).  visible(mwt1gt_2,333). happensAt(active(mwt1gt_2),333).
orientation(mwt1gt_1,160,334). coords(mwt1gt_1,119,150,334).  visible(mwt1gt_1,334). happensAt(active(mwt1gt_1),334). orientation(mwt1gt_2,140,334). coords(mwt1gt_2,128,127,334).  visible(mwt1gt_2,334). happensAt(active(mwt1gt_2),334).
orientation(mwt1gt_1,160,335). coords(mwt1gt_1,119,149,335).  visible(mwt1gt_1,335). happensAt(active(mwt1gt_1),335). orientation(mwt1gt_2,140,335). coords(mwt1gt_2,128,127,335).  visible(mwt1gt_2,335). happensAt(active(mwt1gt_2),335).
orientation(mwt1gt_1,160,336). coords(mwt1gt_1,119,149,336).  visible(mwt1gt_1,336). happensAt(active(mwt1gt_1),336). orientation(mwt1gt_2,140,336). coords(mwt1gt_2,127,127,336).  visible(mwt1gt_2,336). happensAt(active(mwt1gt_2),336).
orientation(mwt1gt_1,160,337). coords(mwt1gt_1,119,149,337).  visible(mwt1gt_1,337). happensAt(active(mwt1gt_1),337). orientation(mwt1gt_2,140,337). coords(mwt1gt_2,127,127,337).  visible(mwt1gt_2,337). happensAt(active(mwt1gt_2),337).
orientation(mwt1gt_1,160,338). coords(mwt1gt_1,119,148,338).  visible(mwt1gt_1,338). happensAt(active(mwt1gt_1),338). orientation(mwt1gt_2,140,338). coords(mwt1gt_2,127,127,338).  visible(mwt1gt_2,338). happensAt(active(mwt1gt_2),338).
orientation(mwt1gt_1,160,339). coords(mwt1gt_1,119,148,339).  visible(mwt1gt_1,339). happensAt(active(mwt1gt_1),339). orientation(mwt1gt_2,140,339). coords(mwt1gt_2,127,127,339).  visible(mwt1gt_2,339). happensAt(active(mwt1gt_2),339).
orientation(mwt1gt_1,160,340). coords(mwt1gt_1,118,148,340).  visible(mwt1gt_1,340). happensAt(active(mwt1gt_1),340). orientation(mwt1gt_2,140,340). coords(mwt1gt_2,127,127,340).  visible(mwt1gt_2,340). happensAt(active(mwt1gt_2),340).
orientation(mwt1gt_1,160,341). coords(mwt1gt_1,118,148,341).  visible(mwt1gt_1,341). happensAt(active(mwt1gt_1),341). orientation(mwt1gt_2,140,341). coords(mwt1gt_2,127,127,341).  visible(mwt1gt_2,341). happensAt(active(mwt1gt_2),341).
orientation(mwt1gt_1,160,342). coords(mwt1gt_1,118,148,342).  visible(mwt1gt_1,342). happensAt(active(mwt1gt_1),342). orientation(mwt1gt_2,140,342). coords(mwt1gt_2,127,127,342).  visible(mwt1gt_2,342). happensAt(active(mwt1gt_2),342).
orientation(mwt1gt_1,160,343). coords(mwt1gt_1,118,148,343).  visible(mwt1gt_1,343). happensAt(active(mwt1gt_1),343). orientation(mwt1gt_2,140,343). coords(mwt1gt_2,126,127,343).  visible(mwt1gt_2,343). happensAt(active(mwt1gt_2),343).
orientation(mwt1gt_1,160,344). coords(mwt1gt_1,118,148,344).  visible(mwt1gt_1,344). happensAt(active(mwt1gt_1),344). orientation(mwt1gt_2,140,344). coords(mwt1gt_2,126,127,344).  visible(mwt1gt_2,344). happensAt(active(mwt1gt_2),344).
orientation(mwt1gt_1,160,345). coords(mwt1gt_1,118,148,345).  visible(mwt1gt_1,345). happensAt(active(mwt1gt_1),345). orientation(mwt1gt_2,140,345). coords(mwt1gt_2,125,127,345).  visible(mwt1gt_2,345). happensAt(active(mwt1gt_2),345).
orientation(mwt1gt_1,160,346). coords(mwt1gt_1,117,148,346).  visible(mwt1gt_1,346). happensAt(active(mwt1gt_1),346). orientation(mwt1gt_2,140,346). coords(mwt1gt_2,125,127,346).  visible(mwt1gt_2,346). happensAt(active(mwt1gt_2),346).
orientation(mwt1gt_1,160,347). coords(mwt1gt_1,117,148,347).  visible(mwt1gt_1,347). happensAt(active(mwt1gt_1),347). orientation(mwt1gt_2,140,347). coords(mwt1gt_2,124,127,347).  visible(mwt1gt_2,347). happensAt(active(mwt1gt_2),347).
orientation(mwt1gt_1,160,348). coords(mwt1gt_1,117,148,348).  visible(mwt1gt_1,348). happensAt(active(mwt1gt_1),348). orientation(mwt1gt_2,140,348). coords(mwt1gt_2,124,127,348).  visible(mwt1gt_2,348). happensAt(active(mwt1gt_2),348).
orientation(mwt1gt_1,160,349). coords(mwt1gt_1,116,148,349).  visible(mwt1gt_1,349). happensAt(active(mwt1gt_1),349). orientation(mwt1gt_2,140,349). coords(mwt1gt_2,124,126,349).  visible(mwt1gt_2,349). happensAt(active(mwt1gt_2),349).
orientation(mwt1gt_1,160,350). coords(mwt1gt_1,116,148,350).  visible(mwt1gt_1,350). happensAt(active(mwt1gt_1),350). orientation(mwt1gt_2,140,350). coords(mwt1gt_2,124,126,350).  visible(mwt1gt_2,350). happensAt(active(mwt1gt_2),350).
orientation(mwt1gt_1,160,351). coords(mwt1gt_1,115,147,351).  visible(mwt1gt_1,351). happensAt(active(mwt1gt_1),351). orientation(mwt1gt_2,140,351). coords(mwt1gt_2,123,126,351).  visible(mwt1gt_2,351). happensAt(active(mwt1gt_2),351).
orientation(mwt1gt_1,160,352). coords(mwt1gt_1,115,147,352).  visible(mwt1gt_1,352). happensAt(active(mwt1gt_1),352). orientation(mwt1gt_2,140,352). coords(mwt1gt_2,121,125,352).  visible(mwt1gt_2,352). happensAt(active(mwt1gt_2),352).
orientation(mwt1gt_1,160,353). coords(mwt1gt_1,115,147,353).  visible(mwt1gt_1,353). happensAt(active(mwt1gt_1),353). orientation(mwt1gt_2,140,353). coords(mwt1gt_2,121,125,353).  visible(mwt1gt_2,353). happensAt(active(mwt1gt_2),353).
#show holdsAt/2.