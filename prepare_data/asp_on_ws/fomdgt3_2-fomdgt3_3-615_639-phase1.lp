
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

	orientation(fomdgt3_2,130,615). coords(fomdgt3_2,253,135,615).  visible(fomdgt3_2,615). happensAt(walking(fomdgt3_2),615). orientation(fomdgt3_3,159,615). coords(fomdgt3_3,69,95,615).  visible(fomdgt3_3,615). happensAt(walking(fomdgt3_3),615).
orientation(fomdgt3_2,130,616). coords(fomdgt3_2,253,135,616).  visible(fomdgt3_2,616). happensAt(walking(fomdgt3_2),616). orientation(fomdgt3_3,159,616). coords(fomdgt3_3,69,95,616).  visible(fomdgt3_3,616). happensAt(walking(fomdgt3_3),616).
orientation(fomdgt3_2,130,617). coords(fomdgt3_2,256,136,617).  visible(fomdgt3_2,617). happensAt(walking(fomdgt3_2),617). orientation(fomdgt3_3,159,617). coords(fomdgt3_3,69,95,617).  visible(fomdgt3_3,617). happensAt(walking(fomdgt3_3),617).
orientation(fomdgt3_2,130,618). coords(fomdgt3_2,256,137,618).  visible(fomdgt3_2,618). happensAt(walking(fomdgt3_2),618). orientation(fomdgt3_3,159,618). coords(fomdgt3_3,71,95,618).  visible(fomdgt3_3,618). happensAt(walking(fomdgt3_3),618).
orientation(fomdgt3_2,130,619). coords(fomdgt3_2,258,139,619).  visible(fomdgt3_2,619). happensAt(walking(fomdgt3_2),619). orientation(fomdgt3_3,159,619). coords(fomdgt3_3,71,95,619).  visible(fomdgt3_3,619). happensAt(walking(fomdgt3_3),619).
orientation(fomdgt3_2,130,620). coords(fomdgt3_2,259,142,620).  visible(fomdgt3_2,620). happensAt(walking(fomdgt3_2),620). orientation(fomdgt3_3,159,620). coords(fomdgt3_3,71,95,620).  visible(fomdgt3_3,620). happensAt(walking(fomdgt3_3),620).
orientation(fomdgt3_2,130,621). coords(fomdgt3_2,260,145,621).  visible(fomdgt3_2,621). happensAt(walking(fomdgt3_2),621). orientation(fomdgt3_3,159,621). coords(fomdgt3_3,71,95,621).  visible(fomdgt3_3,621). happensAt(walking(fomdgt3_3),621).
orientation(fomdgt3_2,130,622). coords(fomdgt3_2,260,148,622).  visible(fomdgt3_2,622). happensAt(walking(fomdgt3_2),622). orientation(fomdgt3_3,159,622). coords(fomdgt3_3,71,95,622).  visible(fomdgt3_3,622). happensAt(walking(fomdgt3_3),622).
orientation(fomdgt3_2,130,623). coords(fomdgt3_2,261,151,623).  visible(fomdgt3_2,623). happensAt(walking(fomdgt3_2),623). orientation(fomdgt3_3,159,623). coords(fomdgt3_3,71,95,623).  visible(fomdgt3_3,623). happensAt(walking(fomdgt3_3),623).
orientation(fomdgt3_2,130,624). coords(fomdgt3_2,262,152,624).  visible(fomdgt3_2,624). happensAt(walking(fomdgt3_2),624). orientation(fomdgt3_3,159,624). coords(fomdgt3_3,71,95,624).  visible(fomdgt3_3,624). happensAt(walking(fomdgt3_3),624).
orientation(fomdgt3_2,130,625). coords(fomdgt3_2,263,154,625).  visible(fomdgt3_2,625). happensAt(walking(fomdgt3_2),625). orientation(fomdgt3_3,159,625). coords(fomdgt3_3,71,95,625).  visible(fomdgt3_3,625). happensAt(walking(fomdgt3_3),625).
orientation(fomdgt3_2,130,626). coords(fomdgt3_2,263,155,626).  visible(fomdgt3_2,626). happensAt(walking(fomdgt3_2),626). orientation(fomdgt3_3,159,626). coords(fomdgt3_3,71,95,626).  visible(fomdgt3_3,626). happensAt(walking(fomdgt3_3),626).
orientation(fomdgt3_2,130,627). coords(fomdgt3_2,263,156,627).  visible(fomdgt3_2,627). happensAt(walking(fomdgt3_2),627). orientation(fomdgt3_3,159,627). coords(fomdgt3_3,71,95,627).  visible(fomdgt3_3,627). happensAt(walking(fomdgt3_3),627).
orientation(fomdgt3_2,130,628). coords(fomdgt3_2,264,157,628).  visible(fomdgt3_2,628). happensAt(walking(fomdgt3_2),628). orientation(fomdgt3_3,159,628). coords(fomdgt3_3,74,95,628).  visible(fomdgt3_3,628). happensAt(walking(fomdgt3_3),628).
orientation(fomdgt3_2,130,629). coords(fomdgt3_2,264,158,629).  visible(fomdgt3_2,629). happensAt(walking(fomdgt3_2),629). orientation(fomdgt3_3,159,629). coords(fomdgt3_3,74,95,629).  visible(fomdgt3_3,629). happensAt(walking(fomdgt3_3),629).
orientation(fomdgt3_2,130,630). coords(fomdgt3_2,265,159,630).  visible(fomdgt3_2,630). happensAt(walking(fomdgt3_2),630). orientation(fomdgt3_3,159,630). coords(fomdgt3_3,74,95,630).  visible(fomdgt3_3,630). happensAt(walking(fomdgt3_3),630).
orientation(fomdgt3_2,130,631). coords(fomdgt3_2,265,160,631).  visible(fomdgt3_2,631). happensAt(walking(fomdgt3_2),631). orientation(fomdgt3_3,159,631). coords(fomdgt3_3,75,95,631).  visible(fomdgt3_3,631). happensAt(walking(fomdgt3_3),631).
orientation(fomdgt3_2,130,632). coords(fomdgt3_2,265,161,632).  visible(fomdgt3_2,632). happensAt(walking(fomdgt3_2),632). orientation(fomdgt3_3,159,632). coords(fomdgt3_3,75,95,632).  visible(fomdgt3_3,632). happensAt(walking(fomdgt3_3),632).
orientation(fomdgt3_2,130,633). coords(fomdgt3_2,265,162,633).  visible(fomdgt3_2,633). happensAt(walking(fomdgt3_2),633). orientation(fomdgt3_3,159,633). coords(fomdgt3_3,75,95,633).  visible(fomdgt3_3,633). happensAt(walking(fomdgt3_3),633).
orientation(fomdgt3_2,130,634). coords(fomdgt3_2,267,163,634).  visible(fomdgt3_2,634). happensAt(walking(fomdgt3_2),634). orientation(fomdgt3_3,159,634). coords(fomdgt3_3,75,95,634).  visible(fomdgt3_3,634). happensAt(walking(fomdgt3_3),634).
orientation(fomdgt3_2,130,635). coords(fomdgt3_2,269,166,635).  visible(fomdgt3_2,635). happensAt(walking(fomdgt3_2),635). orientation(fomdgt3_3,159,635). coords(fomdgt3_3,75,95,635).  visible(fomdgt3_3,635). happensAt(walking(fomdgt3_3),635).
orientation(fomdgt3_2,130,636). coords(fomdgt3_2,271,171,636).  visible(fomdgt3_2,636). happensAt(walking(fomdgt3_2),636). orientation(fomdgt3_3,159,636). coords(fomdgt3_3,75,95,636).  visible(fomdgt3_3,636). happensAt(walking(fomdgt3_3),636).
orientation(fomdgt3_2,130,637). coords(fomdgt3_2,271,174,637).  visible(fomdgt3_2,637). happensAt(walking(fomdgt3_2),637). orientation(fomdgt3_3,159,637). coords(fomdgt3_3,78,95,637).  visible(fomdgt3_3,637). happensAt(walking(fomdgt3_3),637).
orientation(fomdgt3_2,130,638). coords(fomdgt3_2,271,177,638).  visible(fomdgt3_2,638). happensAt(walking(fomdgt3_2),638). orientation(fomdgt3_3,159,638). coords(fomdgt3_3,78,95,638).  visible(fomdgt3_3,638). happensAt(walking(fomdgt3_3),638).
orientation(fomdgt3_2,130,639). coords(fomdgt3_2,272,180,639).  visible(fomdgt3_2,639). happensAt(walking(fomdgt3_2),639). orientation(fomdgt3_3,159,639). coords(fomdgt3_3,80,95,639).  visible(fomdgt3_3,639). happensAt(walking(fomdgt3_3),639).
#show holdsAt/2.