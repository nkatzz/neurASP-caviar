
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

	orientation(bww1gt_1,173,542). coords(bww1gt_1,99,253,542).  visible(bww1gt_1,542). happensAt(walking(bww1gt_1),542). orientation(bww1gt_2,169,542). coords(bww1gt_2,168,212,542).  visible(bww1gt_2,542). happensAt(walking(bww1gt_2),542).
orientation(bww1gt_1,173,543). coords(bww1gt_1,99,252,543).  visible(bww1gt_1,543). happensAt(walking(bww1gt_1),543). orientation(bww1gt_2,169,543). coords(bww1gt_2,168,212,543).  visible(bww1gt_2,543). happensAt(walking(bww1gt_2),543).
orientation(bww1gt_1,173,544). coords(bww1gt_1,99,252,544).  visible(bww1gt_1,544). happensAt(walking(bww1gt_1),544). orientation(bww1gt_2,169,544). coords(bww1gt_2,167,212,544).  visible(bww1gt_2,544). happensAt(walking(bww1gt_2),544).
orientation(bww1gt_1,173,545). coords(bww1gt_1,99,252,545).  visible(bww1gt_1,545). happensAt(walking(bww1gt_1),545). orientation(bww1gt_2,169,545). coords(bww1gt_2,167,212,545).  visible(bww1gt_2,545). happensAt(walking(bww1gt_2),545).
orientation(bww1gt_1,173,546). coords(bww1gt_1,99,252,546).  visible(bww1gt_1,546). happensAt(walking(bww1gt_1),546). orientation(bww1gt_2,169,546). coords(bww1gt_2,166,211,546).  visible(bww1gt_2,546). happensAt(walking(bww1gt_2),546).
orientation(bww1gt_1,173,547). coords(bww1gt_1,99,252,547).  visible(bww1gt_1,547). happensAt(walking(bww1gt_1),547). orientation(bww1gt_2,169,547). coords(bww1gt_2,166,211,547).  visible(bww1gt_2,547). happensAt(walking(bww1gt_2),547).
orientation(bww1gt_1,173,548). coords(bww1gt_1,99,250,548).  visible(bww1gt_1,548). happensAt(walking(bww1gt_1),548). orientation(bww1gt_2,169,548). coords(bww1gt_2,166,210,548).  visible(bww1gt_2,548). happensAt(walking(bww1gt_2),548).
orientation(bww1gt_1,173,549). coords(bww1gt_1,99,250,549).  visible(bww1gt_1,549). happensAt(walking(bww1gt_1),549). orientation(bww1gt_2,169,549). coords(bww1gt_2,165,209,549).  visible(bww1gt_2,549). happensAt(walking(bww1gt_2),549).
orientation(bww1gt_1,173,550). coords(bww1gt_1,99,250,550).  visible(bww1gt_1,550). happensAt(walking(bww1gt_1),550). orientation(bww1gt_2,169,550). coords(bww1gt_2,165,208,550).  visible(bww1gt_2,550). happensAt(walking(bww1gt_2),550).
orientation(bww1gt_1,173,551). coords(bww1gt_1,99,250,551).  visible(bww1gt_1,551). happensAt(walking(bww1gt_1),551). orientation(bww1gt_2,169,551). coords(bww1gt_2,164,208,551).  visible(bww1gt_2,551). happensAt(walking(bww1gt_2),551).
orientation(bww1gt_1,173,552). coords(bww1gt_1,99,249,552).  visible(bww1gt_1,552). happensAt(inactive(bww1gt_1),552). orientation(bww1gt_2,169,552). coords(bww1gt_2,163,207,552).  visible(bww1gt_2,552). happensAt(walking(bww1gt_2),552).
orientation(bww1gt_1,173,553). coords(bww1gt_1,99,249,553).  visible(bww1gt_1,553). happensAt(inactive(bww1gt_1),553). orientation(bww1gt_2,169,553). coords(bww1gt_2,162,207,553).  visible(bww1gt_2,553). happensAt(walking(bww1gt_2),553).
orientation(bww1gt_1,173,554). coords(bww1gt_1,99,249,554).  visible(bww1gt_1,554). happensAt(inactive(bww1gt_1),554). orientation(bww1gt_2,169,554). coords(bww1gt_2,160,206,554).  visible(bww1gt_2,554). happensAt(walking(bww1gt_2),554).
orientation(bww1gt_1,173,555). coords(bww1gt_1,99,249,555).  visible(bww1gt_1,555). happensAt(inactive(bww1gt_1),555). orientation(bww1gt_2,169,555). coords(bww1gt_2,158,206,555).  visible(bww1gt_2,555). happensAt(walking(bww1gt_2),555).
orientation(bww1gt_1,173,556). coords(bww1gt_1,99,249,556).  visible(bww1gt_1,556). happensAt(inactive(bww1gt_1),556). orientation(bww1gt_2,169,556). coords(bww1gt_2,158,205,556).  visible(bww1gt_2,556). happensAt(walking(bww1gt_2),556).
orientation(bww1gt_1,173,557). coords(bww1gt_1,99,249,557).  visible(bww1gt_1,557). happensAt(inactive(bww1gt_1),557). orientation(bww1gt_2,169,557). coords(bww1gt_2,155,205,557).  visible(bww1gt_2,557). happensAt(walking(bww1gt_2),557).
orientation(bww1gt_1,173,558). coords(bww1gt_1,99,249,558).  visible(bww1gt_1,558). happensAt(inactive(bww1gt_1),558). orientation(bww1gt_2,169,558). coords(bww1gt_2,154,205,558).  visible(bww1gt_2,558). happensAt(walking(bww1gt_2),558).
orientation(bww1gt_1,173,559). coords(bww1gt_1,99,249,559).  visible(bww1gt_1,559). happensAt(inactive(bww1gt_1),559). orientation(bww1gt_2,169,559). coords(bww1gt_2,153,205,559).  visible(bww1gt_2,559). happensAt(walking(bww1gt_2),559).
orientation(bww1gt_1,173,560). coords(bww1gt_1,99,249,560).  visible(bww1gt_1,560). happensAt(inactive(bww1gt_1),560). orientation(bww1gt_2,169,560). coords(bww1gt_2,153,205,560).  visible(bww1gt_2,560). happensAt(walking(bww1gt_2),560).
orientation(bww1gt_1,173,561). coords(bww1gt_1,99,249,561).  visible(bww1gt_1,561). happensAt(inactive(bww1gt_1),561). orientation(bww1gt_2,169,561). coords(bww1gt_2,152,205,561).  visible(bww1gt_2,561). happensAt(walking(bww1gt_2),561).
orientation(bww1gt_1,173,562). coords(bww1gt_1,99,249,562).  visible(bww1gt_1,562). happensAt(inactive(bww1gt_1),562). orientation(bww1gt_2,169,562). coords(bww1gt_2,152,206,562).  visible(bww1gt_2,562). happensAt(walking(bww1gt_2),562).
orientation(bww1gt_1,173,563). coords(bww1gt_1,99,248,563).  visible(bww1gt_1,563). happensAt(inactive(bww1gt_1),563). orientation(bww1gt_2,169,563). coords(bww1gt_2,152,206,563).  visible(bww1gt_2,563). happensAt(walking(bww1gt_2),563).
orientation(bww1gt_1,173,564). coords(bww1gt_1,99,248,564).  visible(bww1gt_1,564). happensAt(inactive(bww1gt_1),564). orientation(bww1gt_2,169,564). coords(bww1gt_2,152,206,564).  visible(bww1gt_2,564). happensAt(walking(bww1gt_2),564).
orientation(bww1gt_1,173,565). coords(bww1gt_1,99,248,565).  visible(bww1gt_1,565). happensAt(inactive(bww1gt_1),565). orientation(bww1gt_2,169,565). coords(bww1gt_2,151,206,565).  visible(bww1gt_2,565). happensAt(walking(bww1gt_2),565).
orientation(bww1gt_1,173,566). coords(bww1gt_1,99,248,566).  visible(bww1gt_1,566). happensAt(inactive(bww1gt_1),566). orientation(bww1gt_2,169,566). coords(bww1gt_2,151,206,566).  visible(bww1gt_2,566). happensAt(walking(bww1gt_2),566).
#show holdsAt/2.