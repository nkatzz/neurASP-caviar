
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

	orientation(wk2gt_1,122,523). coords(wk2gt_1,115,53,523).  visible(wk2gt_1,523). happensAt(inactive(wk2gt_1),523). orientation(wk2gt_4,150,523). coords(wk2gt_4,58,97,523).  visible(wk2gt_4,523). happensAt(walking(wk2gt_4),523).
orientation(wk2gt_1,122,524). coords(wk2gt_1,115,53,524).  visible(wk2gt_1,524). happensAt(inactive(wk2gt_1),524). orientation(wk2gt_4,150,524). coords(wk2gt_4,57,97,524).  visible(wk2gt_4,524). happensAt(walking(wk2gt_4),524).
orientation(wk2gt_1,122,525). coords(wk2gt_1,115,53,525).  visible(wk2gt_1,525). happensAt(inactive(wk2gt_1),525). orientation(wk2gt_4,150,525). coords(wk2gt_4,57,97,525).  visible(wk2gt_4,525). happensAt(walking(wk2gt_4),525).
orientation(wk2gt_1,122,526). coords(wk2gt_1,115,53,526).  visible(wk2gt_1,526). happensAt(inactive(wk2gt_1),526). orientation(wk2gt_4,150,526). coords(wk2gt_4,56,97,526).  visible(wk2gt_4,526). happensAt(walking(wk2gt_4),526).
orientation(wk2gt_1,122,527). coords(wk2gt_1,115,53,527).  visible(wk2gt_1,527). happensAt(inactive(wk2gt_1),527). orientation(wk2gt_4,150,527). coords(wk2gt_4,55,97,527).  visible(wk2gt_4,527). happensAt(walking(wk2gt_4),527).
orientation(wk2gt_1,122,528). coords(wk2gt_1,115,53,528).  visible(wk2gt_1,528). happensAt(inactive(wk2gt_1),528). orientation(wk2gt_4,150,528). coords(wk2gt_4,55,97,528).  visible(wk2gt_4,528). happensAt(walking(wk2gt_4),528).
orientation(wk2gt_1,122,529). coords(wk2gt_1,115,53,529).  visible(wk2gt_1,529). happensAt(inactive(wk2gt_1),529). orientation(wk2gt_4,150,529). coords(wk2gt_4,55,100,529).  visible(wk2gt_4,529). happensAt(walking(wk2gt_4),529).
orientation(wk2gt_1,122,530). coords(wk2gt_1,115,53,530).  visible(wk2gt_1,530). happensAt(inactive(wk2gt_1),530). orientation(wk2gt_4,150,530). coords(wk2gt_4,55,100,530).  visible(wk2gt_4,530). happensAt(walking(wk2gt_4),530).
orientation(wk2gt_1,122,531). coords(wk2gt_1,115,53,531).  visible(wk2gt_1,531). happensAt(inactive(wk2gt_1),531). orientation(wk2gt_4,150,531). coords(wk2gt_4,55,100,531).  visible(wk2gt_4,531). happensAt(walking(wk2gt_4),531).
orientation(wk2gt_1,122,532). coords(wk2gt_1,115,53,532).  visible(wk2gt_1,532). happensAt(inactive(wk2gt_1),532). orientation(wk2gt_4,150,532). coords(wk2gt_4,55,101,532).  visible(wk2gt_4,532). happensAt(walking(wk2gt_4),532).
orientation(wk2gt_1,122,533). coords(wk2gt_1,115,53,533).  visible(wk2gt_1,533). happensAt(inactive(wk2gt_1),533). orientation(wk2gt_4,150,533). coords(wk2gt_4,54,101,533).  visible(wk2gt_4,533). happensAt(walking(wk2gt_4),533).
orientation(wk2gt_1,122,534). coords(wk2gt_1,115,53,534).  visible(wk2gt_1,534). happensAt(inactive(wk2gt_1),534). orientation(wk2gt_4,150,534). coords(wk2gt_4,54,101,534).  visible(wk2gt_4,534). happensAt(walking(wk2gt_4),534).
orientation(wk2gt_1,122,535). coords(wk2gt_1,115,53,535).  visible(wk2gt_1,535). happensAt(inactive(wk2gt_1),535). orientation(wk2gt_4,150,535). coords(wk2gt_4,53,101,535).  visible(wk2gt_4,535). happensAt(walking(wk2gt_4),535).
orientation(wk2gt_1,122,536). coords(wk2gt_1,115,53,536).  visible(wk2gt_1,536). happensAt(inactive(wk2gt_1),536). orientation(wk2gt_4,150,536). coords(wk2gt_4,53,101,536).  visible(wk2gt_4,536). happensAt(walking(wk2gt_4),536).
orientation(wk2gt_1,122,537). coords(wk2gt_1,115,53,537).  visible(wk2gt_1,537). happensAt(inactive(wk2gt_1),537). orientation(wk2gt_4,150,537). coords(wk2gt_4,50,101,537).  visible(wk2gt_4,537). happensAt(walking(wk2gt_4),537).
orientation(wk2gt_1,122,538). coords(wk2gt_1,115,53,538).  visible(wk2gt_1,538). happensAt(inactive(wk2gt_1),538). orientation(wk2gt_4,150,538). coords(wk2gt_4,50,101,538).  visible(wk2gt_4,538). happensAt(walking(wk2gt_4),538).
orientation(wk2gt_1,122,539). coords(wk2gt_1,115,53,539).  visible(wk2gt_1,539). happensAt(inactive(wk2gt_1),539). orientation(wk2gt_4,150,539). coords(wk2gt_4,50,101,539).  visible(wk2gt_4,539). happensAt(walking(wk2gt_4),539).
orientation(wk2gt_1,122,540). coords(wk2gt_1,115,53,540).  visible(wk2gt_1,540). happensAt(inactive(wk2gt_1),540). orientation(wk2gt_4,150,540). coords(wk2gt_4,50,101,540).  visible(wk2gt_4,540). happensAt(walking(wk2gt_4),540).
orientation(wk2gt_1,122,541). coords(wk2gt_1,115,53,541).  visible(wk2gt_1,541). happensAt(inactive(wk2gt_1),541). orientation(wk2gt_4,150,541). coords(wk2gt_4,50,101,541).  visible(wk2gt_4,541). happensAt(walking(wk2gt_4),541).
orientation(wk2gt_1,122,542). coords(wk2gt_1,115,53,542).  visible(wk2gt_1,542). happensAt(inactive(wk2gt_1),542). orientation(wk2gt_4,150,542). coords(wk2gt_4,49,101,542).  visible(wk2gt_4,542). happensAt(walking(wk2gt_4),542).
orientation(wk2gt_1,122,543). coords(wk2gt_1,115,53,543).  visible(wk2gt_1,543). happensAt(inactive(wk2gt_1),543). orientation(wk2gt_4,150,543). coords(wk2gt_4,49,101,543).  visible(wk2gt_4,543). happensAt(walking(wk2gt_4),543).
orientation(wk2gt_1,122,544). coords(wk2gt_1,115,53,544).  visible(wk2gt_1,544). happensAt(inactive(wk2gt_1),544). orientation(wk2gt_4,149,544). coords(wk2gt_4,49,101,544).  visible(wk2gt_4,544). happensAt(walking(wk2gt_4),544).
orientation(wk2gt_1,122,545). coords(wk2gt_1,115,53,545).  visible(wk2gt_1,545). happensAt(inactive(wk2gt_1),545). orientation(wk2gt_4,149,545). coords(wk2gt_4,49,101,545).  visible(wk2gt_4,545). happensAt(walking(wk2gt_4),545).
orientation(wk2gt_1,122,546). coords(wk2gt_1,115,53,546).  visible(wk2gt_1,546). happensAt(inactive(wk2gt_1),546). orientation(wk2gt_4,149,546). coords(wk2gt_4,49,101,546).  visible(wk2gt_4,546). happensAt(walking(wk2gt_4),546).
orientation(wk2gt_1,122,547). coords(wk2gt_1,115,53,547).  visible(wk2gt_1,547). happensAt(inactive(wk2gt_1),547). orientation(wk2gt_4,149,547). coords(wk2gt_4,49,101,547).  visible(wk2gt_4,547). happensAt(walking(wk2gt_4),547).
#show holdsAt/2.