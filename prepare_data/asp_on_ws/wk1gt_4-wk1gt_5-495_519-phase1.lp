
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

	orientation(wk1gt_4,157,495). coords(wk1gt_4,56,170,495).  visible(wk1gt_4,495). happensAt(active(wk1gt_4),495). orientation(wk1gt_5,162,495). coords(wk1gt_5,57,150,495).  visible(wk1gt_5,495). happensAt(active(wk1gt_5),495).
orientation(wk1gt_4,157,496). coords(wk1gt_4,56,170,496).  visible(wk1gt_4,496). happensAt(active(wk1gt_4),496). orientation(wk1gt_5,162,496). coords(wk1gt_5,57,150,496).  visible(wk1gt_5,496). happensAt(active(wk1gt_5),496).
orientation(wk1gt_4,157,497). coords(wk1gt_4,56,170,497).  visible(wk1gt_4,497). happensAt(active(wk1gt_4),497). orientation(wk1gt_5,162,497). coords(wk1gt_5,57,150,497).  visible(wk1gt_5,497). happensAt(active(wk1gt_5),497).
orientation(wk1gt_4,157,498). coords(wk1gt_4,56,170,498).  visible(wk1gt_4,498). happensAt(active(wk1gt_4),498). orientation(wk1gt_5,162,498). coords(wk1gt_5,57,150,498).  visible(wk1gt_5,498). happensAt(inactive(wk1gt_5),498).
orientation(wk1gt_4,157,499). coords(wk1gt_4,56,170,499).  visible(wk1gt_4,499). happensAt(active(wk1gt_4),499). orientation(wk1gt_5,162,499). coords(wk1gt_5,57,150,499).  visible(wk1gt_5,499). happensAt(inactive(wk1gt_5),499).
orientation(wk1gt_4,157,500). coords(wk1gt_4,56,170,500).  visible(wk1gt_4,500). happensAt(active(wk1gt_4),500). orientation(wk1gt_5,162,500). coords(wk1gt_5,57,150,500).  visible(wk1gt_5,500). happensAt(inactive(wk1gt_5),500).
orientation(wk1gt_4,157,501). coords(wk1gt_4,56,170,501).  visible(wk1gt_4,501). happensAt(active(wk1gt_4),501). orientation(wk1gt_5,162,501). coords(wk1gt_5,57,150,501).  visible(wk1gt_5,501). happensAt(inactive(wk1gt_5),501).
orientation(wk1gt_4,157,502). coords(wk1gt_4,56,170,502).  visible(wk1gt_4,502). happensAt(active(wk1gt_4),502). orientation(wk1gt_5,162,502). coords(wk1gt_5,57,150,502).  visible(wk1gt_5,502). happensAt(inactive(wk1gt_5),502).
orientation(wk1gt_4,157,503). coords(wk1gt_4,56,170,503).  visible(wk1gt_4,503). happensAt(active(wk1gt_4),503). orientation(wk1gt_5,162,503). coords(wk1gt_5,57,150,503).  visible(wk1gt_5,503). happensAt(inactive(wk1gt_5),503).
orientation(wk1gt_4,157,504). coords(wk1gt_4,56,170,504).  visible(wk1gt_4,504). happensAt(active(wk1gt_4),504). orientation(wk1gt_5,162,504). coords(wk1gt_5,57,150,504).  visible(wk1gt_5,504). happensAt(inactive(wk1gt_5),504).
orientation(wk1gt_4,157,505). coords(wk1gt_4,56,170,505).  visible(wk1gt_4,505). happensAt(active(wk1gt_4),505). orientation(wk1gt_5,162,505). coords(wk1gt_5,57,150,505).  visible(wk1gt_5,505). happensAt(inactive(wk1gt_5),505).
orientation(wk1gt_4,157,506). coords(wk1gt_4,56,170,506).  visible(wk1gt_4,506). happensAt(active(wk1gt_4),506). orientation(wk1gt_5,162,506). coords(wk1gt_5,57,150,506).  visible(wk1gt_5,506). happensAt(inactive(wk1gt_5),506).
orientation(wk1gt_4,157,507). coords(wk1gt_4,56,170,507).  visible(wk1gt_4,507). happensAt(active(wk1gt_4),507). orientation(wk1gt_5,162,507). coords(wk1gt_5,57,150,507).  visible(wk1gt_5,507). happensAt(inactive(wk1gt_5),507).
orientation(wk1gt_4,157,508). coords(wk1gt_4,56,170,508).  visible(wk1gt_4,508). happensAt(active(wk1gt_4),508). orientation(wk1gt_5,162,508). coords(wk1gt_5,57,150,508).  visible(wk1gt_5,508). happensAt(inactive(wk1gt_5),508).
orientation(wk1gt_4,157,509). coords(wk1gt_4,56,170,509).  visible(wk1gt_4,509). happensAt(active(wk1gt_4),509). orientation(wk1gt_5,162,509). coords(wk1gt_5,57,150,509).  visible(wk1gt_5,509). happensAt(inactive(wk1gt_5),509).
orientation(wk1gt_4,157,510). coords(wk1gt_4,56,170,510).  visible(wk1gt_4,510). happensAt(active(wk1gt_4),510). orientation(wk1gt_5,162,510). coords(wk1gt_5,57,150,510).  visible(wk1gt_5,510). happensAt(inactive(wk1gt_5),510).
orientation(wk1gt_4,157,511). coords(wk1gt_4,56,170,511).  visible(wk1gt_4,511). happensAt(active(wk1gt_4),511). orientation(wk1gt_5,162,511). coords(wk1gt_5,57,150,511).  visible(wk1gt_5,511). happensAt(inactive(wk1gt_5),511).
orientation(wk1gt_4,157,512). coords(wk1gt_4,56,170,512).  visible(wk1gt_4,512). happensAt(active(wk1gt_4),512). orientation(wk1gt_5,162,512). coords(wk1gt_5,57,150,512).  visible(wk1gt_5,512). happensAt(inactive(wk1gt_5),512).
orientation(wk1gt_4,157,513). coords(wk1gt_4,56,170,513).  visible(wk1gt_4,513). happensAt(active(wk1gt_4),513). orientation(wk1gt_5,162,513). coords(wk1gt_5,57,150,513).  visible(wk1gt_5,513). happensAt(inactive(wk1gt_5),513).
orientation(wk1gt_4,157,514). coords(wk1gt_4,56,169,514).  visible(wk1gt_4,514). happensAt(active(wk1gt_4),514). orientation(wk1gt_5,162,514). coords(wk1gt_5,57,150,514).  visible(wk1gt_5,514). happensAt(inactive(wk1gt_5),514).
orientation(wk1gt_4,157,515). coords(wk1gt_4,56,169,515).  visible(wk1gt_4,515). happensAt(active(wk1gt_4),515). orientation(wk1gt_5,162,515). coords(wk1gt_5,57,150,515).  visible(wk1gt_5,515). happensAt(inactive(wk1gt_5),515).
orientation(wk1gt_4,157,516). coords(wk1gt_4,56,169,516).  visible(wk1gt_4,516). happensAt(active(wk1gt_4),516). orientation(wk1gt_5,162,516). coords(wk1gt_5,57,150,516).  visible(wk1gt_5,516). happensAt(inactive(wk1gt_5),516).
orientation(wk1gt_4,157,517). coords(wk1gt_4,56,169,517).  visible(wk1gt_4,517). happensAt(active(wk1gt_4),517). orientation(wk1gt_5,162,517). coords(wk1gt_5,57,150,517).  visible(wk1gt_5,517). happensAt(inactive(wk1gt_5),517).
orientation(wk1gt_4,157,518). coords(wk1gt_4,56,169,518).  visible(wk1gt_4,518). happensAt(inactive(wk1gt_4),518). orientation(wk1gt_5,162,518). coords(wk1gt_5,57,150,518).  visible(wk1gt_5,518). happensAt(inactive(wk1gt_5),518).
orientation(wk1gt_4,157,519). coords(wk1gt_4,56,169,519).  visible(wk1gt_4,519). happensAt(inactive(wk1gt_4),519). orientation(wk1gt_5,162,519). coords(wk1gt_5,57,150,519).  visible(wk1gt_5,519). happensAt(inactive(wk1gt_5),519).
#show holdsAt/2.