
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

	orientation(mwt2gt_1,143,289). coords(mwt2gt_1,138,75,289).  visible(mwt2gt_1,289). happensAt(inactive(mwt2gt_1),289). orientation(mwt2gt_2,145,289). coords(mwt2gt_2,161,106,289).  visible(mwt2gt_2,289). happensAt(walking(mwt2gt_2),289).
orientation(mwt2gt_1,143,290). coords(mwt2gt_1,138,75,290).  visible(mwt2gt_1,290). happensAt(inactive(mwt2gt_1),290). orientation(mwt2gt_2,145,290). coords(mwt2gt_2,160,106,290).  visible(mwt2gt_2,290). happensAt(walking(mwt2gt_2),290).
orientation(mwt2gt_1,143,291). coords(mwt2gt_1,138,75,291).  visible(mwt2gt_1,291). happensAt(inactive(mwt2gt_1),291). orientation(mwt2gt_2,145,291). coords(mwt2gt_2,159,104,291).  visible(mwt2gt_2,291). happensAt(walking(mwt2gt_2),291).
orientation(mwt2gt_1,143,292). coords(mwt2gt_1,138,75,292).  visible(mwt2gt_1,292). happensAt(inactive(mwt2gt_1),292). orientation(mwt2gt_2,145,292). coords(mwt2gt_2,158,103,292).  visible(mwt2gt_2,292). happensAt(walking(mwt2gt_2),292).
orientation(mwt2gt_1,143,293). coords(mwt2gt_1,138,75,293).  visible(mwt2gt_1,293). happensAt(inactive(mwt2gt_1),293). orientation(mwt2gt_2,145,293). coords(mwt2gt_2,157,102,293).  visible(mwt2gt_2,293). happensAt(walking(mwt2gt_2),293).
orientation(mwt2gt_1,143,294). coords(mwt2gt_1,138,75,294).  visible(mwt2gt_1,294). happensAt(inactive(mwt2gt_1),294). orientation(mwt2gt_2,145,294). coords(mwt2gt_2,155,101,294).  visible(mwt2gt_2,294). happensAt(walking(mwt2gt_2),294).
orientation(mwt2gt_1,143,295). coords(mwt2gt_1,138,75,295).  visible(mwt2gt_1,295). happensAt(inactive(mwt2gt_1),295). orientation(mwt2gt_2,145,295). coords(mwt2gt_2,153,100,295).  visible(mwt2gt_2,295). happensAt(walking(mwt2gt_2),295).
orientation(mwt2gt_1,143,296). coords(mwt2gt_1,138,75,296).  visible(mwt2gt_1,296). happensAt(inactive(mwt2gt_1),296). orientation(mwt2gt_2,145,296). coords(mwt2gt_2,153,100,296).  visible(mwt2gt_2,296). happensAt(walking(mwt2gt_2),296).
orientation(mwt2gt_1,143,297). coords(mwt2gt_1,138,75,297).  visible(mwt2gt_1,297). happensAt(inactive(mwt2gt_1),297). orientation(mwt2gt_2,145,297). coords(mwt2gt_2,151,100,297).  visible(mwt2gt_2,297). happensAt(walking(mwt2gt_2),297).
orientation(mwt2gt_1,143,298). coords(mwt2gt_1,138,75,298).  visible(mwt2gt_1,298). happensAt(inactive(mwt2gt_1),298). orientation(mwt2gt_2,145,298). coords(mwt2gt_2,151,99,298).  visible(mwt2gt_2,298). happensAt(walking(mwt2gt_2),298).
orientation(mwt2gt_1,143,299). coords(mwt2gt_1,138,75,299).  visible(mwt2gt_1,299). happensAt(inactive(mwt2gt_1),299). orientation(mwt2gt_2,145,299). coords(mwt2gt_2,151,99,299).  visible(mwt2gt_2,299). happensAt(walking(mwt2gt_2),299).
orientation(mwt2gt_1,143,300). coords(mwt2gt_1,138,75,300).  visible(mwt2gt_1,300). happensAt(inactive(mwt2gt_1),300). orientation(mwt2gt_2,145,300). coords(mwt2gt_2,151,99,300).  visible(mwt2gt_2,300). happensAt(walking(mwt2gt_2),300).
orientation(mwt2gt_1,143,301). coords(mwt2gt_1,138,75,301).  visible(mwt2gt_1,301). happensAt(inactive(mwt2gt_1),301). orientation(mwt2gt_2,145,301). coords(mwt2gt_2,151,98,301).  visible(mwt2gt_2,301). happensAt(walking(mwt2gt_2),301).
orientation(mwt2gt_1,143,302). coords(mwt2gt_1,138,75,302).  visible(mwt2gt_1,302). happensAt(inactive(mwt2gt_1),302). orientation(mwt2gt_2,142,302). coords(mwt2gt_2,151,98,302).  visible(mwt2gt_2,302). happensAt(walking(mwt2gt_2),302).
orientation(mwt2gt_1,143,303). coords(mwt2gt_1,138,75,303).  visible(mwt2gt_1,303). happensAt(inactive(mwt2gt_1),303). orientation(mwt2gt_2,142,303). coords(mwt2gt_2,150,97,303).  visible(mwt2gt_2,303). happensAt(walking(mwt2gt_2),303).
orientation(mwt2gt_1,143,304). coords(mwt2gt_1,138,75,304).  visible(mwt2gt_1,304). happensAt(inactive(mwt2gt_1),304). orientation(mwt2gt_2,142,304). coords(mwt2gt_2,150,96,304).  visible(mwt2gt_2,304). happensAt(walking(mwt2gt_2),304).
orientation(mwt2gt_1,143,305). coords(mwt2gt_1,138,75,305).  visible(mwt2gt_1,305). happensAt(inactive(mwt2gt_1),305). orientation(mwt2gt_2,142,305). coords(mwt2gt_2,148,95,305).  visible(mwt2gt_2,305). happensAt(walking(mwt2gt_2),305).
orientation(mwt2gt_1,143,306). coords(mwt2gt_1,138,75,306).  visible(mwt2gt_1,306). happensAt(inactive(mwt2gt_1),306). orientation(mwt2gt_2,142,306). coords(mwt2gt_2,147,93,306).  visible(mwt2gt_2,306). happensAt(walking(mwt2gt_2),306).
orientation(mwt2gt_1,143,307). coords(mwt2gt_1,138,75,307).  visible(mwt2gt_1,307). happensAt(inactive(mwt2gt_1),307). orientation(mwt2gt_2,142,307). coords(mwt2gt_2,146,92,307).  visible(mwt2gt_2,307). happensAt(walking(mwt2gt_2),307).
orientation(mwt2gt_1,143,308). coords(mwt2gt_1,138,75,308).  visible(mwt2gt_1,308). happensAt(inactive(mwt2gt_1),308). orientation(mwt2gt_2,142,308). coords(mwt2gt_2,146,91,308).  visible(mwt2gt_2,308). happensAt(active(mwt2gt_2),308).
orientation(mwt2gt_1,143,309). coords(mwt2gt_1,138,75,309).  visible(mwt2gt_1,309). happensAt(inactive(mwt2gt_1),309). orientation(mwt2gt_2,142,309). coords(mwt2gt_2,146,91,309).  visible(mwt2gt_2,309). happensAt(active(mwt2gt_2),309).
orientation(mwt2gt_1,143,310). coords(mwt2gt_1,138,75,310).  visible(mwt2gt_1,310). happensAt(inactive(mwt2gt_1),310). orientation(mwt2gt_2,142,310). coords(mwt2gt_2,145,89,310).  visible(mwt2gt_2,310). happensAt(active(mwt2gt_2),310).
orientation(mwt2gt_1,143,311). coords(mwt2gt_1,138,75,311).  visible(mwt2gt_1,311). happensAt(inactive(mwt2gt_1),311). orientation(mwt2gt_2,142,311). coords(mwt2gt_2,145,89,311).  visible(mwt2gt_2,311). happensAt(active(mwt2gt_2),311).
orientation(mwt2gt_1,143,312). coords(mwt2gt_1,138,75,312).  visible(mwt2gt_1,312). happensAt(inactive(mwt2gt_1),312). orientation(mwt2gt_2,144,312). coords(mwt2gt_2,145,89,312).  visible(mwt2gt_2,312). happensAt(active(mwt2gt_2),312).
orientation(mwt2gt_1,143,313). coords(mwt2gt_1,138,75,313).  visible(mwt2gt_1,313). happensAt(inactive(mwt2gt_1),313). orientation(mwt2gt_2,146,313). coords(mwt2gt_2,144,89,313).  visible(mwt2gt_2,313). happensAt(active(mwt2gt_2),313).
#show holdsAt/2.