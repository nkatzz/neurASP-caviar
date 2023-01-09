
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

	orientation(ms3ggt_0,168,524). coords(ms3ggt_0,73,191,524).  visible(ms3ggt_0,524). happensAt(inactive(ms3ggt_0),524). orientation(ms3ggt_2,141,524). coords(ms3ggt_2,86,109,524).  visible(ms3ggt_2,524). happensAt(walking(ms3ggt_2),524).
orientation(ms3ggt_0,168,525). coords(ms3ggt_0,73,191,525).  visible(ms3ggt_0,525). happensAt(inactive(ms3ggt_0),525). orientation(ms3ggt_2,141,525). coords(ms3ggt_2,86,108,525).  visible(ms3ggt_2,525). happensAt(walking(ms3ggt_2),525).
orientation(ms3ggt_0,168,526). coords(ms3ggt_0,73,191,526).  visible(ms3ggt_0,526). happensAt(inactive(ms3ggt_0),526). orientation(ms3ggt_2,141,526). coords(ms3ggt_2,86,108,526).  visible(ms3ggt_2,526). happensAt(walking(ms3ggt_2),526).
orientation(ms3ggt_0,168,527). coords(ms3ggt_0,73,191,527).  visible(ms3ggt_0,527). happensAt(inactive(ms3ggt_0),527). orientation(ms3ggt_2,141,527). coords(ms3ggt_2,86,106,527).  visible(ms3ggt_2,527). happensAt(walking(ms3ggt_2),527).
orientation(ms3ggt_0,168,528). coords(ms3ggt_0,73,191,528).  visible(ms3ggt_0,528). happensAt(inactive(ms3ggt_0),528). orientation(ms3ggt_2,141,528). coords(ms3ggt_2,86,106,528).  visible(ms3ggt_2,528). happensAt(walking(ms3ggt_2),528).
orientation(ms3ggt_0,168,529). coords(ms3ggt_0,73,191,529).  visible(ms3ggt_0,529). happensAt(inactive(ms3ggt_0),529). orientation(ms3ggt_2,141,529). coords(ms3ggt_2,86,106,529).  visible(ms3ggt_2,529). happensAt(walking(ms3ggt_2),529).
orientation(ms3ggt_0,168,530). coords(ms3ggt_0,73,191,530).  visible(ms3ggt_0,530). happensAt(inactive(ms3ggt_0),530). orientation(ms3ggt_2,141,530). coords(ms3ggt_2,86,107,530).  visible(ms3ggt_2,530). happensAt(walking(ms3ggt_2),530).
orientation(ms3ggt_0,168,531). coords(ms3ggt_0,73,191,531).  visible(ms3ggt_0,531). happensAt(inactive(ms3ggt_0),531). orientation(ms3ggt_2,141,531). coords(ms3ggt_2,86,106,531).  visible(ms3ggt_2,531). happensAt(walking(ms3ggt_2),531).
orientation(ms3ggt_0,168,532). coords(ms3ggt_0,73,191,532).  visible(ms3ggt_0,532). happensAt(inactive(ms3ggt_0),532). orientation(ms3ggt_2,141,532). coords(ms3ggt_2,86,106,532).  visible(ms3ggt_2,532). happensAt(walking(ms3ggt_2),532).
orientation(ms3ggt_0,168,533). coords(ms3ggt_0,73,191,533).  visible(ms3ggt_0,533). happensAt(inactive(ms3ggt_0),533). orientation(ms3ggt_2,141,533). coords(ms3ggt_2,85,104,533).  visible(ms3ggt_2,533). happensAt(walking(ms3ggt_2),533).
orientation(ms3ggt_0,168,534). coords(ms3ggt_0,73,191,534).  visible(ms3ggt_0,534). happensAt(inactive(ms3ggt_0),534). orientation(ms3ggt_2,141,534). coords(ms3ggt_2,85,103,534).  visible(ms3ggt_2,534). happensAt(walking(ms3ggt_2),534).
orientation(ms3ggt_0,168,535). coords(ms3ggt_0,73,191,535).  visible(ms3ggt_0,535). happensAt(inactive(ms3ggt_0),535). orientation(ms3ggt_2,151,535). coords(ms3ggt_2,85,103,535).  visible(ms3ggt_2,535). happensAt(walking(ms3ggt_2),535).
orientation(ms3ggt_0,168,536). coords(ms3ggt_0,73,191,536).  visible(ms3ggt_0,536). happensAt(inactive(ms3ggt_0),536). orientation(ms3ggt_2,151,536). coords(ms3ggt_2,85,103,536).  visible(ms3ggt_2,536). happensAt(walking(ms3ggt_2),536).
orientation(ms3ggt_0,168,537). coords(ms3ggt_0,73,191,537).  visible(ms3ggt_0,537). happensAt(inactive(ms3ggt_0),537). orientation(ms3ggt_2,151,537). coords(ms3ggt_2,85,102,537).  visible(ms3ggt_2,537). happensAt(walking(ms3ggt_2),537).
orientation(ms3ggt_0,168,538). coords(ms3ggt_0,73,191,538).  visible(ms3ggt_0,538). happensAt(inactive(ms3ggt_0),538). orientation(ms3ggt_2,151,538). coords(ms3ggt_2,85,102,538).  visible(ms3ggt_2,538). happensAt(walking(ms3ggt_2),538).
orientation(ms3ggt_0,168,539). coords(ms3ggt_0,73,191,539).  visible(ms3ggt_0,539). happensAt(inactive(ms3ggt_0),539). orientation(ms3ggt_2,151,539). coords(ms3ggt_2,85,102,539).  visible(ms3ggt_2,539). happensAt(walking(ms3ggt_2),539).
orientation(ms3ggt_0,168,540). coords(ms3ggt_0,73,191,540).  visible(ms3ggt_0,540). happensAt(inactive(ms3ggt_0),540). orientation(ms3ggt_2,151,540). coords(ms3ggt_2,85,102,540).  visible(ms3ggt_2,540). happensAt(walking(ms3ggt_2),540).
orientation(ms3ggt_0,168,541). coords(ms3ggt_0,73,191,541).  visible(ms3ggt_0,541). happensAt(inactive(ms3ggt_0),541). orientation(ms3ggt_2,139,541). coords(ms3ggt_2,85,102,541).  visible(ms3ggt_2,541). happensAt(walking(ms3ggt_2),541).
orientation(ms3ggt_0,168,542). coords(ms3ggt_0,73,191,542).  visible(ms3ggt_0,542). happensAt(inactive(ms3ggt_0),542). orientation(ms3ggt_2,139,542). coords(ms3ggt_2,84,101,542).  visible(ms3ggt_2,542). happensAt(walking(ms3ggt_2),542).
orientation(ms3ggt_0,168,543). coords(ms3ggt_0,73,191,543).  visible(ms3ggt_0,543). happensAt(inactive(ms3ggt_0),543). orientation(ms3ggt_2,139,543). coords(ms3ggt_2,84,100,543).  visible(ms3ggt_2,543). happensAt(walking(ms3ggt_2),543).
orientation(ms3ggt_0,168,544). coords(ms3ggt_0,73,191,544).  visible(ms3ggt_0,544). happensAt(inactive(ms3ggt_0),544). orientation(ms3ggt_2,139,544). coords(ms3ggt_2,84,100,544).  visible(ms3ggt_2,544). happensAt(walking(ms3ggt_2),544).
orientation(ms3ggt_0,168,545). coords(ms3ggt_0,73,191,545).  visible(ms3ggt_0,545). happensAt(inactive(ms3ggt_0),545). orientation(ms3ggt_2,139,545). coords(ms3ggt_2,84,99,545).  visible(ms3ggt_2,545). happensAt(walking(ms3ggt_2),545).
orientation(ms3ggt_0,168,546). coords(ms3ggt_0,73,191,546).  visible(ms3ggt_0,546). happensAt(inactive(ms3ggt_0),546). orientation(ms3ggt_2,139,546). coords(ms3ggt_2,84,99,546).  visible(ms3ggt_2,546). happensAt(walking(ms3ggt_2),546).
orientation(ms3ggt_0,168,547). coords(ms3ggt_0,73,191,547).  visible(ms3ggt_0,547). happensAt(inactive(ms3ggt_0),547). orientation(ms3ggt_2,142,547). coords(ms3ggt_2,84,98,547).  visible(ms3ggt_2,547). happensAt(walking(ms3ggt_2),547).
orientation(ms3ggt_0,168,548). coords(ms3ggt_0,73,191,548).  visible(ms3ggt_0,548). happensAt(inactive(ms3ggt_0),548). orientation(ms3ggt_2,142,548). coords(ms3ggt_2,84,96,548).  visible(ms3ggt_2,548). happensAt(walking(ms3ggt_2),548).
#show holdsAt/2.