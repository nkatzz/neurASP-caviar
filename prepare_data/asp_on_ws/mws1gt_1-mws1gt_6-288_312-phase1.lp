
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

	orientation(mws1gt_1,155,288). coords(mws1gt_1,88,105,288).  visible(mws1gt_1,288). happensAt(walking(mws1gt_1),288). orientation(mws1gt_6,146,288). coords(mws1gt_6,38,146,288).  visible(mws1gt_6,288). happensAt(walking(mws1gt_6),288).
orientation(mws1gt_1,155,289). coords(mws1gt_1,88,105,289).  visible(mws1gt_1,289). happensAt(walking(mws1gt_1),289). orientation(mws1gt_6,146,289). coords(mws1gt_6,37,146,289).  visible(mws1gt_6,289). happensAt(walking(mws1gt_6),289).
orientation(mws1gt_1,155,290). coords(mws1gt_1,87,104,290).  visible(mws1gt_1,290). happensAt(walking(mws1gt_1),290). orientation(mws1gt_6,146,290). coords(mws1gt_6,37,147,290).  visible(mws1gt_6,290). happensAt(walking(mws1gt_6),290).
orientation(mws1gt_1,155,291). coords(mws1gt_1,87,104,291).  visible(mws1gt_1,291). happensAt(walking(mws1gt_1),291). orientation(mws1gt_6,146,291). coords(mws1gt_6,37,148,291).  visible(mws1gt_6,291). happensAt(walking(mws1gt_6),291).
orientation(mws1gt_1,155,292). coords(mws1gt_1,87,104,292).  visible(mws1gt_1,292). happensAt(walking(mws1gt_1),292). orientation(mws1gt_6,146,292). coords(mws1gt_6,36,150,292).  visible(mws1gt_6,292). happensAt(walking(mws1gt_6),292).
orientation(mws1gt_1,155,293). coords(mws1gt_1,87,104,293).  visible(mws1gt_1,293). happensAt(walking(mws1gt_1),293). orientation(mws1gt_6,146,293). coords(mws1gt_6,36,151,293).  visible(mws1gt_6,293). happensAt(walking(mws1gt_6),293).
orientation(mws1gt_1,155,294). coords(mws1gt_1,87,103,294).  visible(mws1gt_1,294). happensAt(walking(mws1gt_1),294). orientation(mws1gt_6,146,294). coords(mws1gt_6,36,152,294).  visible(mws1gt_6,294). happensAt(walking(mws1gt_6),294).
orientation(mws1gt_1,155,295). coords(mws1gt_1,87,103,295).  visible(mws1gt_1,295). happensAt(walking(mws1gt_1),295). orientation(mws1gt_6,146,295). coords(mws1gt_6,36,152,295).  visible(mws1gt_6,295). happensAt(walking(mws1gt_6),295).
orientation(mws1gt_1,155,296). coords(mws1gt_1,87,103,296).  visible(mws1gt_1,296). happensAt(walking(mws1gt_1),296). orientation(mws1gt_6,146,296). coords(mws1gt_6,36,153,296).  visible(mws1gt_6,296). happensAt(walking(mws1gt_6),296).
orientation(mws1gt_1,155,297). coords(mws1gt_1,86,103,297).  visible(mws1gt_1,297). happensAt(walking(mws1gt_1),297). orientation(mws1gt_6,146,297). coords(mws1gt_6,36,153,297).  visible(mws1gt_6,297). happensAt(walking(mws1gt_6),297).
orientation(mws1gt_1,155,298). coords(mws1gt_1,86,103,298).  visible(mws1gt_1,298). happensAt(walking(mws1gt_1),298). orientation(mws1gt_6,146,298). coords(mws1gt_6,36,154,298).  visible(mws1gt_6,298). happensAt(walking(mws1gt_6),298).
orientation(mws1gt_1,155,299). coords(mws1gt_1,85,104,299).  visible(mws1gt_1,299). happensAt(walking(mws1gt_1),299). orientation(mws1gt_6,146,299). coords(mws1gt_6,36,155,299).  visible(mws1gt_6,299). happensAt(walking(mws1gt_6),299).
orientation(mws1gt_1,155,300). coords(mws1gt_1,85,104,300).  visible(mws1gt_1,300). happensAt(walking(mws1gt_1),300). orientation(mws1gt_6,146,300). coords(mws1gt_6,36,155,300).  visible(mws1gt_6,300). happensAt(walking(mws1gt_6),300).
orientation(mws1gt_1,155,301). coords(mws1gt_1,84,103,301).  visible(mws1gt_1,301). happensAt(walking(mws1gt_1),301). orientation(mws1gt_6,146,301). coords(mws1gt_6,36,155,301).  visible(mws1gt_6,301). happensAt(walking(mws1gt_6),301).
orientation(mws1gt_1,155,302). coords(mws1gt_1,83,103,302).  visible(mws1gt_1,302). happensAt(walking(mws1gt_1),302). orientation(mws1gt_6,146,302). coords(mws1gt_6,36,155,302).  visible(mws1gt_6,302). happensAt(walking(mws1gt_6),302).
orientation(mws1gt_1,155,303). coords(mws1gt_1,82,103,303).  visible(mws1gt_1,303). happensAt(walking(mws1gt_1),303). orientation(mws1gt_6,146,303). coords(mws1gt_6,36,156,303).  visible(mws1gt_6,303). happensAt(walking(mws1gt_6),303).
orientation(mws1gt_1,155,304). coords(mws1gt_1,82,103,304).  visible(mws1gt_1,304). happensAt(walking(mws1gt_1),304). orientation(mws1gt_6,146,304). coords(mws1gt_6,35,157,304).  visible(mws1gt_6,304). happensAt(walking(mws1gt_6),304).
orientation(mws1gt_1,155,305). coords(mws1gt_1,81,103,305).  visible(mws1gt_1,305). happensAt(walking(mws1gt_1),305). orientation(mws1gt_6,146,305). coords(mws1gt_6,35,159,305).  visible(mws1gt_6,305). happensAt(walking(mws1gt_6),305).
orientation(mws1gt_1,155,306). coords(mws1gt_1,81,103,306).  visible(mws1gt_1,306). happensAt(walking(mws1gt_1),306). orientation(mws1gt_6,146,306). coords(mws1gt_6,35,160,306).  visible(mws1gt_6,306). happensAt(walking(mws1gt_6),306).
orientation(mws1gt_1,155,307). coords(mws1gt_1,80,103,307).  visible(mws1gt_1,307). happensAt(walking(mws1gt_1),307). orientation(mws1gt_6,146,307). coords(mws1gt_6,35,162,307).  visible(mws1gt_6,307). happensAt(walking(mws1gt_6),307).
orientation(mws1gt_1,155,308). coords(mws1gt_1,80,103,308).  visible(mws1gt_1,308). happensAt(walking(mws1gt_1),308). orientation(mws1gt_6,146,308). coords(mws1gt_6,35,163,308).  visible(mws1gt_6,308). happensAt(walking(mws1gt_6),308).
orientation(mws1gt_1,155,309). coords(mws1gt_1,80,103,309).  visible(mws1gt_1,309). happensAt(walking(mws1gt_1),309). orientation(mws1gt_6,146,309). coords(mws1gt_6,35,164,309).  visible(mws1gt_6,309). happensAt(walking(mws1gt_6),309).
orientation(mws1gt_1,155,310). coords(mws1gt_1,80,103,310).  visible(mws1gt_1,310). happensAt(walking(mws1gt_1),310). orientation(mws1gt_6,146,310). coords(mws1gt_6,35,164,310).  visible(mws1gt_6,310). happensAt(walking(mws1gt_6),310).
orientation(mws1gt_1,155,311). coords(mws1gt_1,80,103,311).  visible(mws1gt_1,311). happensAt(walking(mws1gt_1),311). orientation(mws1gt_6,146,311). coords(mws1gt_6,35,165,311).  visible(mws1gt_6,311). happensAt(walking(mws1gt_6),311).
orientation(mws1gt_1,155,312). coords(mws1gt_1,80,102,312).  visible(mws1gt_1,312). happensAt(walking(mws1gt_1),312). orientation(mws1gt_6,146,312). coords(mws1gt_6,35,166,312).  visible(mws1gt_6,312). happensAt(walking(mws1gt_6),312).
#show holdsAt/2.