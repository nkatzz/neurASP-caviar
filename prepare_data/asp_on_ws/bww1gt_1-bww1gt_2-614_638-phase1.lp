
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

	orientation(bww1gt_1,173,614). coords(bww1gt_1,101,251,614).  visible(bww1gt_1,614). happensAt(walking(bww1gt_1),614). orientation(bww1gt_2,2,614). coords(bww1gt_2,140,209,614).  visible(bww1gt_2,614). happensAt(active(bww1gt_2),614).
orientation(bww1gt_1,173,615). coords(bww1gt_1,101,252,615).  visible(bww1gt_1,615). happensAt(walking(bww1gt_1),615). orientation(bww1gt_2,4,615). coords(bww1gt_2,140,208,615).  visible(bww1gt_2,615). happensAt(active(bww1gt_2),615).
orientation(bww1gt_1,173,616). coords(bww1gt_1,101,252,616).  visible(bww1gt_1,616). happensAt(walking(bww1gt_1),616). orientation(bww1gt_2,3,616). coords(bww1gt_2,141,207,616).  visible(bww1gt_2,616). happensAt(active(bww1gt_2),616).
orientation(bww1gt_1,173,617). coords(bww1gt_1,101,253,617).  visible(bww1gt_1,617). happensAt(walking(bww1gt_1),617). orientation(bww1gt_2,2,617). coords(bww1gt_2,142,206,617).  visible(bww1gt_2,617). happensAt(active(bww1gt_2),617).
orientation(bww1gt_1,173,618). coords(bww1gt_1,101,253,618).  visible(bww1gt_1,618). happensAt(walking(bww1gt_1),618). orientation(bww1gt_2,1,618). coords(bww1gt_2,142,206,618).  visible(bww1gt_2,618). happensAt(active(bww1gt_2),618).
orientation(bww1gt_1,173,619). coords(bww1gt_1,101,253,619).  visible(bww1gt_1,619). happensAt(walking(bww1gt_1),619). orientation(bww1gt_2,3,619). coords(bww1gt_2,143,207,619).  visible(bww1gt_2,619). happensAt(active(bww1gt_2),619).
orientation(bww1gt_1,173,620). coords(bww1gt_1,101,253,620).  visible(bww1gt_1,620). happensAt(walking(bww1gt_1),620). orientation(bww1gt_2,3,620). coords(bww1gt_2,144,206,620).  visible(bww1gt_2,620). happensAt(walking(bww1gt_2),620).
orientation(bww1gt_1,173,621). coords(bww1gt_1,101,254,621).  visible(bww1gt_1,621). happensAt(walking(bww1gt_1),621). orientation(bww1gt_2,178,621). coords(bww1gt_2,144,207,621).  visible(bww1gt_2,621). happensAt(walking(bww1gt_2),621).
orientation(bww1gt_1,173,622). coords(bww1gt_1,101,254,622).  visible(bww1gt_1,622). happensAt(walking(bww1gt_1),622). orientation(bww1gt_2,178,622). coords(bww1gt_2,145,207,622).  visible(bww1gt_2,622). happensAt(walking(bww1gt_2),622).
orientation(bww1gt_1,173,623). coords(bww1gt_1,101,254,623).  visible(bww1gt_1,623). happensAt(walking(bww1gt_1),623). orientation(bww1gt_2,177,623). coords(bww1gt_2,146,207,623).  visible(bww1gt_2,623). happensAt(walking(bww1gt_2),623).
orientation(bww1gt_1,173,624). coords(bww1gt_1,101,254,624).  visible(bww1gt_1,624). happensAt(walking(bww1gt_1),624). orientation(bww1gt_2,178,624). coords(bww1gt_2,146,207,624).  visible(bww1gt_2,624). happensAt(walking(bww1gt_2),624).
orientation(bww1gt_1,173,625). coords(bww1gt_1,101,254,625).  visible(bww1gt_1,625). happensAt(walking(bww1gt_1),625). orientation(bww1gt_2,169,625). coords(bww1gt_2,149,207,625).  visible(bww1gt_2,625). happensAt(walking(bww1gt_2),625).
orientation(bww1gt_1,173,626). coords(bww1gt_1,101,255,626).  visible(bww1gt_1,626). happensAt(walking(bww1gt_1),626). orientation(bww1gt_2,169,626). coords(bww1gt_2,152,208,626).  visible(bww1gt_2,626). happensAt(walking(bww1gt_2),626).
orientation(bww1gt_1,173,627). coords(bww1gt_1,101,255,627).  visible(bww1gt_1,627). happensAt(walking(bww1gt_1),627). orientation(bww1gt_2,169,627). coords(bww1gt_2,155,209,627).  visible(bww1gt_2,627). happensAt(walking(bww1gt_2),627).
orientation(bww1gt_1,173,628). coords(bww1gt_1,101,255,628).  visible(bww1gt_1,628). happensAt(walking(bww1gt_1),628). orientation(bww1gt_2,169,628). coords(bww1gt_2,157,209,628).  visible(bww1gt_2,628). happensAt(walking(bww1gt_2),628).
orientation(bww1gt_1,173,629). coords(bww1gt_1,102,256,629).  visible(bww1gt_1,629). happensAt(walking(bww1gt_1),629). orientation(bww1gt_2,169,629). coords(bww1gt_2,160,211,629).  visible(bww1gt_2,629). happensAt(walking(bww1gt_2),629).
orientation(bww1gt_1,173,630). coords(bww1gt_1,102,256,630).  visible(bww1gt_1,630). happensAt(walking(bww1gt_1),630). orientation(bww1gt_2,169,630). coords(bww1gt_2,164,211,630).  visible(bww1gt_2,630). happensAt(walking(bww1gt_2),630).
orientation(bww1gt_1,173,631). coords(bww1gt_1,102,256,631).  visible(bww1gt_1,631). happensAt(walking(bww1gt_1),631). orientation(bww1gt_2,169,631). coords(bww1gt_2,164,211,631).  visible(bww1gt_2,631). happensAt(walking(bww1gt_2),631).
orientation(bww1gt_1,173,632). coords(bww1gt_1,102,256,632).  visible(bww1gt_1,632). happensAt(walking(bww1gt_1),632). orientation(bww1gt_2,169,632). coords(bww1gt_2,165,211,632).  visible(bww1gt_2,632). happensAt(walking(bww1gt_2),632).
orientation(bww1gt_1,173,633). coords(bww1gt_1,102,256,633).  visible(bww1gt_1,633). happensAt(walking(bww1gt_1),633). orientation(bww1gt_2,169,633). coords(bww1gt_2,167,212,633).  visible(bww1gt_2,633). happensAt(walking(bww1gt_2),633).
orientation(bww1gt_1,173,634). coords(bww1gt_1,102,256,634).  visible(bww1gt_1,634). happensAt(walking(bww1gt_1),634). orientation(bww1gt_2,169,634). coords(bww1gt_2,167,212,634).  visible(bww1gt_2,634). happensAt(walking(bww1gt_2),634).
orientation(bww1gt_1,173,635). coords(bww1gt_1,102,257,635).  visible(bww1gt_1,635). happensAt(walking(bww1gt_1),635). orientation(bww1gt_2,169,635). coords(bww1gt_2,168,213,635).  visible(bww1gt_2,635). happensAt(walking(bww1gt_2),635).
orientation(bww1gt_1,173,636). coords(bww1gt_1,102,257,636).  visible(bww1gt_1,636). happensAt(walking(bww1gt_1),636). orientation(bww1gt_2,169,636). coords(bww1gt_2,169,215,636).  visible(bww1gt_2,636). happensAt(walking(bww1gt_2),636).
orientation(bww1gt_1,173,637). coords(bww1gt_1,102,257,637).  visible(bww1gt_1,637). happensAt(walking(bww1gt_1),637). orientation(bww1gt_2,169,637). coords(bww1gt_2,169,215,637).  visible(bww1gt_2,637). happensAt(walking(bww1gt_2),637).
orientation(bww1gt_1,173,638). coords(bww1gt_1,102,257,638).  visible(bww1gt_1,638). happensAt(walking(bww1gt_1),638). orientation(bww1gt_2,169,638). coords(bww1gt_2,170,216,638).  visible(bww1gt_2,638). happensAt(walking(bww1gt_2),638).
#show holdsAt/2.