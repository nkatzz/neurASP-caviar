
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

	orientation(wk2gt_1,122,492). coords(wk2gt_1,115,53,492).  visible(wk2gt_1,492). happensAt(inactive(wk2gt_1),492). orientation(wk2gt_2,141,492). coords(wk2gt_2,99,63,492).  visible(wk2gt_2,492). happensAt(inactive(wk2gt_2),492).
orientation(wk2gt_1,122,493). coords(wk2gt_1,115,53,493).  visible(wk2gt_1,493). happensAt(inactive(wk2gt_1),493). orientation(wk2gt_2,141,493). coords(wk2gt_2,99,63,493).  visible(wk2gt_2,493). happensAt(inactive(wk2gt_2),493).
orientation(wk2gt_1,122,494). coords(wk2gt_1,115,53,494).  visible(wk2gt_1,494). happensAt(inactive(wk2gt_1),494). orientation(wk2gt_2,141,494). coords(wk2gt_2,99,63,494).  visible(wk2gt_2,494). happensAt(inactive(wk2gt_2),494).
orientation(wk2gt_1,122,495). coords(wk2gt_1,115,53,495).  visible(wk2gt_1,495). happensAt(inactive(wk2gt_1),495). orientation(wk2gt_2,141,495). coords(wk2gt_2,100,63,495).  visible(wk2gt_2,495). happensAt(inactive(wk2gt_2),495).
orientation(wk2gt_1,122,496). coords(wk2gt_1,115,53,496).  visible(wk2gt_1,496). happensAt(inactive(wk2gt_1),496). orientation(wk2gt_2,141,496). coords(wk2gt_2,99,63,496).  visible(wk2gt_2,496). happensAt(inactive(wk2gt_2),496).
orientation(wk2gt_1,122,497). coords(wk2gt_1,115,53,497).  visible(wk2gt_1,497). happensAt(inactive(wk2gt_1),497). orientation(wk2gt_2,141,497). coords(wk2gt_2,99,63,497).  visible(wk2gt_2,497). happensAt(inactive(wk2gt_2),497).
orientation(wk2gt_1,122,498). coords(wk2gt_1,115,53,498).  visible(wk2gt_1,498). happensAt(inactive(wk2gt_1),498). orientation(wk2gt_2,141,498). coords(wk2gt_2,99,63,498).  visible(wk2gt_2,498). happensAt(inactive(wk2gt_2),498).
orientation(wk2gt_1,122,499). coords(wk2gt_1,115,53,499).  visible(wk2gt_1,499). happensAt(inactive(wk2gt_1),499). orientation(wk2gt_2,141,499). coords(wk2gt_2,99,63,499).  visible(wk2gt_2,499). happensAt(inactive(wk2gt_2),499).
orientation(wk2gt_1,122,500). coords(wk2gt_1,115,53,500).  visible(wk2gt_1,500). happensAt(inactive(wk2gt_1),500). orientation(wk2gt_2,141,500). coords(wk2gt_2,99,63,500).  visible(wk2gt_2,500). happensAt(inactive(wk2gt_2),500).
orientation(wk2gt_1,122,501). coords(wk2gt_1,115,53,501).  visible(wk2gt_1,501). happensAt(inactive(wk2gt_1),501). orientation(wk2gt_2,141,501). coords(wk2gt_2,99,63,501).  visible(wk2gt_2,501). happensAt(inactive(wk2gt_2),501).
orientation(wk2gt_1,122,502). coords(wk2gt_1,115,53,502).  visible(wk2gt_1,502). happensAt(inactive(wk2gt_1),502). orientation(wk2gt_2,141,502). coords(wk2gt_2,99,63,502).  visible(wk2gt_2,502). happensAt(inactive(wk2gt_2),502).
orientation(wk2gt_1,122,503). coords(wk2gt_1,115,53,503).  visible(wk2gt_1,503). happensAt(inactive(wk2gt_1),503). orientation(wk2gt_2,141,503). coords(wk2gt_2,99,63,503).  visible(wk2gt_2,503). happensAt(inactive(wk2gt_2),503).
orientation(wk2gt_1,122,504). coords(wk2gt_1,115,53,504).  visible(wk2gt_1,504). happensAt(inactive(wk2gt_1),504). orientation(wk2gt_2,141,504). coords(wk2gt_2,99,63,504).  visible(wk2gt_2,504). happensAt(inactive(wk2gt_2),504).
orientation(wk2gt_1,122,505). coords(wk2gt_1,115,53,505).  visible(wk2gt_1,505). happensAt(inactive(wk2gt_1),505). orientation(wk2gt_2,141,505). coords(wk2gt_2,99,63,505).  visible(wk2gt_2,505). happensAt(inactive(wk2gt_2),505).
orientation(wk2gt_1,122,506). coords(wk2gt_1,115,53,506).  visible(wk2gt_1,506). happensAt(inactive(wk2gt_1),506). orientation(wk2gt_2,141,506). coords(wk2gt_2,99,63,506).  visible(wk2gt_2,506). happensAt(inactive(wk2gt_2),506).
orientation(wk2gt_1,122,507). coords(wk2gt_1,115,53,507).  visible(wk2gt_1,507). happensAt(inactive(wk2gt_1),507). orientation(wk2gt_2,141,507). coords(wk2gt_2,99,63,507).  visible(wk2gt_2,507). happensAt(inactive(wk2gt_2),507).
orientation(wk2gt_1,122,508). coords(wk2gt_1,115,53,508).  visible(wk2gt_1,508). happensAt(inactive(wk2gt_1),508). orientation(wk2gt_2,141,508). coords(wk2gt_2,99,63,508).  visible(wk2gt_2,508). happensAt(inactive(wk2gt_2),508).
orientation(wk2gt_1,122,509). coords(wk2gt_1,115,53,509).  visible(wk2gt_1,509). happensAt(inactive(wk2gt_1),509). orientation(wk2gt_2,141,509). coords(wk2gt_2,99,63,509).  visible(wk2gt_2,509). happensAt(inactive(wk2gt_2),509).
orientation(wk2gt_1,122,510). coords(wk2gt_1,115,53,510).  visible(wk2gt_1,510). happensAt(inactive(wk2gt_1),510). orientation(wk2gt_2,141,510). coords(wk2gt_2,99,63,510).  visible(wk2gt_2,510). happensAt(inactive(wk2gt_2),510).
orientation(wk2gt_1,122,511). coords(wk2gt_1,115,53,511).  visible(wk2gt_1,511). happensAt(inactive(wk2gt_1),511). orientation(wk2gt_2,141,511). coords(wk2gt_2,99,63,511).  visible(wk2gt_2,511). happensAt(inactive(wk2gt_2),511).
orientation(wk2gt_1,122,512). coords(wk2gt_1,115,53,512).  visible(wk2gt_1,512). happensAt(inactive(wk2gt_1),512). orientation(wk2gt_2,141,512). coords(wk2gt_2,99,63,512).  visible(wk2gt_2,512). happensAt(inactive(wk2gt_2),512).
orientation(wk2gt_1,122,513). coords(wk2gt_1,115,53,513).  visible(wk2gt_1,513). happensAt(inactive(wk2gt_1),513). orientation(wk2gt_2,141,513). coords(wk2gt_2,99,63,513).  visible(wk2gt_2,513). happensAt(inactive(wk2gt_2),513).
orientation(wk2gt_1,122,514). coords(wk2gt_1,115,53,514).  visible(wk2gt_1,514). happensAt(inactive(wk2gt_1),514). orientation(wk2gt_2,141,514). coords(wk2gt_2,99,63,514).  visible(wk2gt_2,514). happensAt(inactive(wk2gt_2),514).
orientation(wk2gt_1,122,515). coords(wk2gt_1,115,53,515).  visible(wk2gt_1,515). happensAt(inactive(wk2gt_1),515). orientation(wk2gt_2,141,515). coords(wk2gt_2,99,63,515).  visible(wk2gt_2,515). happensAt(inactive(wk2gt_2),515).
orientation(wk2gt_1,122,516). coords(wk2gt_1,115,53,516).  visible(wk2gt_1,516). happensAt(inactive(wk2gt_1),516). orientation(wk2gt_2,141,516). coords(wk2gt_2,99,63,516).  visible(wk2gt_2,516). happensAt(inactive(wk2gt_2),516).
#show holdsAt/2.