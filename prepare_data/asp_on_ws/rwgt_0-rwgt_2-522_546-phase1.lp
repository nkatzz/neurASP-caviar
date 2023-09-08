
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

	orientation(rwgt_0,137,522). coords(rwgt_0,316,283,522).  visible(rwgt_0,522). happensAt(inactive(rwgt_0),522). orientation(rwgt_2,136,522). coords(rwgt_2,160,32,522).  visible(rwgt_2,522). happensAt(walking(rwgt_2),522).
orientation(rwgt_0,137,523). coords(rwgt_0,316,283,523).  visible(rwgt_0,523). happensAt(inactive(rwgt_0),523). orientation(rwgt_2,136,523). coords(rwgt_2,159,33,523).  visible(rwgt_2,523). happensAt(walking(rwgt_2),523).
orientation(rwgt_0,137,524). coords(rwgt_0,316,283,524).  visible(rwgt_0,524). happensAt(inactive(rwgt_0),524). orientation(rwgt_2,132,524). coords(rwgt_2,159,33,524).  visible(rwgt_2,524). happensAt(walking(rwgt_2),524).
orientation(rwgt_0,137,525). coords(rwgt_0,316,283,525).  visible(rwgt_0,525). happensAt(inactive(rwgt_0),525). orientation(rwgt_2,132,525). coords(rwgt_2,159,33,525).  visible(rwgt_2,525). happensAt(walking(rwgt_2),525).
orientation(rwgt_0,137,526). coords(rwgt_0,316,283,526).  visible(rwgt_0,526). happensAt(inactive(rwgt_0),526). orientation(rwgt_2,132,526). coords(rwgt_2,159,34,526).  visible(rwgt_2,526). happensAt(walking(rwgt_2),526).
orientation(rwgt_0,137,527). coords(rwgt_0,316,283,527).  visible(rwgt_0,527). happensAt(inactive(rwgt_0),527). orientation(rwgt_2,132,527). coords(rwgt_2,157,34,527).  visible(rwgt_2,527). happensAt(walking(rwgt_2),527).
orientation(rwgt_0,137,528). coords(rwgt_0,316,283,528).  visible(rwgt_0,528). happensAt(inactive(rwgt_0),528). orientation(rwgt_2,132,528). coords(rwgt_2,157,33,528).  visible(rwgt_2,528). happensAt(walking(rwgt_2),528).
orientation(rwgt_0,137,529). coords(rwgt_0,316,283,529).  visible(rwgt_0,529). happensAt(inactive(rwgt_0),529). orientation(rwgt_2,132,529). coords(rwgt_2,157,33,529).  visible(rwgt_2,529). happensAt(walking(rwgt_2),529).
orientation(rwgt_0,137,530). coords(rwgt_0,316,283,530).  visible(rwgt_0,530). happensAt(inactive(rwgt_0),530). orientation(rwgt_2,132,530). coords(rwgt_2,156,34,530).  visible(rwgt_2,530). happensAt(walking(rwgt_2),530).
orientation(rwgt_0,137,531). coords(rwgt_0,316,283,531).  visible(rwgt_0,531). happensAt(inactive(rwgt_0),531). orientation(rwgt_2,132,531). coords(rwgt_2,156,34,531).  visible(rwgt_2,531). happensAt(walking(rwgt_2),531).
orientation(rwgt_0,137,532). coords(rwgt_0,316,283,532).  visible(rwgt_0,532). happensAt(inactive(rwgt_0),532). orientation(rwgt_2,132,532). coords(rwgt_2,155,34,532).  visible(rwgt_2,532). happensAt(walking(rwgt_2),532).
orientation(rwgt_0,137,533). coords(rwgt_0,316,283,533).  visible(rwgt_0,533). happensAt(inactive(rwgt_0),533). orientation(rwgt_2,132,533). coords(rwgt_2,155,35,533).  visible(rwgt_2,533). happensAt(walking(rwgt_2),533).
orientation(rwgt_0,137,534). coords(rwgt_0,316,283,534).  visible(rwgt_0,534). happensAt(inactive(rwgt_0),534). orientation(rwgt_2,132,534). coords(rwgt_2,155,35,534).  visible(rwgt_2,534). happensAt(walking(rwgt_2),534).
orientation(rwgt_0,137,535). coords(rwgt_0,316,283,535).  visible(rwgt_0,535). happensAt(inactive(rwgt_0),535). orientation(rwgt_2,132,535). coords(rwgt_2,156,35,535).  visible(rwgt_2,535). happensAt(walking(rwgt_2),535).
orientation(rwgt_0,137,536). coords(rwgt_0,316,283,536).  visible(rwgt_0,536). happensAt(inactive(rwgt_0),536). orientation(rwgt_2,132,536). coords(rwgt_2,156,35,536).  visible(rwgt_2,536). happensAt(walking(rwgt_2),536).
orientation(rwgt_0,137,537). coords(rwgt_0,316,283,537).  visible(rwgt_0,537). happensAt(inactive(rwgt_0),537). orientation(rwgt_2,132,537). coords(rwgt_2,156,35,537).  visible(rwgt_2,537). happensAt(walking(rwgt_2),537).
orientation(rwgt_0,137,538). coords(rwgt_0,316,283,538).  visible(rwgt_0,538). happensAt(inactive(rwgt_0),538). orientation(rwgt_2,132,538). coords(rwgt_2,155,35,538).  visible(rwgt_2,538). happensAt(walking(rwgt_2),538).
orientation(rwgt_0,137,539). coords(rwgt_0,316,283,539).  visible(rwgt_0,539). happensAt(inactive(rwgt_0),539). orientation(rwgt_2,137,539). coords(rwgt_2,155,35,539).  visible(rwgt_2,539). happensAt(walking(rwgt_2),539).
orientation(rwgt_0,137,540). coords(rwgt_0,316,283,540).  visible(rwgt_0,540). happensAt(inactive(rwgt_0),540). orientation(rwgt_2,137,540). coords(rwgt_2,155,35,540).  visible(rwgt_2,540). happensAt(walking(rwgt_2),540).
orientation(rwgt_0,137,541). coords(rwgt_0,316,283,541).  visible(rwgt_0,541). happensAt(inactive(rwgt_0),541). orientation(rwgt_2,137,541). coords(rwgt_2,155,35,541).  visible(rwgt_2,541). happensAt(walking(rwgt_2),541).
orientation(rwgt_0,137,542). coords(rwgt_0,316,283,542).  visible(rwgt_0,542). happensAt(inactive(rwgt_0),542). orientation(rwgt_2,137,542). coords(rwgt_2,155,35,542).  visible(rwgt_2,542). happensAt(walking(rwgt_2),542).
orientation(rwgt_0,137,543). coords(rwgt_0,316,283,543).  visible(rwgt_0,543). happensAt(inactive(rwgt_0),543). orientation(rwgt_2,137,543). coords(rwgt_2,155,35,543).  visible(rwgt_2,543). happensAt(walking(rwgt_2),543).
orientation(rwgt_0,137,544). coords(rwgt_0,316,283,544).  visible(rwgt_0,544). happensAt(inactive(rwgt_0),544). orientation(rwgt_2,137,544). coords(rwgt_2,155,36,544).  visible(rwgt_2,544). happensAt(walking(rwgt_2),544).
orientation(rwgt_0,137,545). coords(rwgt_0,316,283,545).  visible(rwgt_0,545). happensAt(inactive(rwgt_0),545). orientation(rwgt_2,137,545). coords(rwgt_2,154,37,545).  visible(rwgt_2,545). happensAt(walking(rwgt_2),545).
orientation(rwgt_0,137,546). coords(rwgt_0,316,283,546).  visible(rwgt_0,546). happensAt(inactive(rwgt_0),546). orientation(rwgt_2,137,546). coords(rwgt_2,154,37,546).  visible(rwgt_2,546). happensAt(walking(rwgt_2),546).
#show holdsAt/2.