
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

	orientation(lbpugt_2,132,623). coords(lbpugt_2,181,50,623).  visible(lbpugt_2,623). happensAt(active(lbpugt_2),623). orientation(lbpugt_4,135,623). coords(lbpugt_4,219,34,623).  visible(lbpugt_4,623). happensAt(inactive(lbpugt_4),623).
orientation(lbpugt_2,132,624). coords(lbpugt_2,181,50,624).  visible(lbpugt_2,624). happensAt(active(lbpugt_2),624). orientation(lbpugt_4,135,624). coords(lbpugt_4,219,34,624).  visible(lbpugt_4,624). happensAt(inactive(lbpugt_4),624).
orientation(lbpugt_2,132,625). coords(lbpugt_2,181,50,625).  visible(lbpugt_2,625). happensAt(active(lbpugt_2),625). orientation(lbpugt_4,135,625). coords(lbpugt_4,219,34,625).  visible(lbpugt_4,625). happensAt(inactive(lbpugt_4),625).
orientation(lbpugt_2,132,626). coords(lbpugt_2,181,50,626).  visible(lbpugt_2,626). happensAt(active(lbpugt_2),626). orientation(lbpugt_4,135,626). coords(lbpugt_4,219,34,626).  visible(lbpugt_4,626). happensAt(inactive(lbpugt_4),626).
orientation(lbpugt_2,132,627). coords(lbpugt_2,181,50,627).  visible(lbpugt_2,627). happensAt(active(lbpugt_2),627). orientation(lbpugt_4,135,627). coords(lbpugt_4,219,34,627).  visible(lbpugt_4,627). happensAt(inactive(lbpugt_4),627).
orientation(lbpugt_2,132,628). coords(lbpugt_2,181,50,628).  visible(lbpugt_2,628). happensAt(active(lbpugt_2),628). orientation(lbpugt_4,135,628). coords(lbpugt_4,219,34,628).  visible(lbpugt_4,628). happensAt(inactive(lbpugt_4),628).
orientation(lbpugt_2,132,629). coords(lbpugt_2,181,50,629).  visible(lbpugt_2,629). happensAt(active(lbpugt_2),629). orientation(lbpugt_4,135,629). coords(lbpugt_4,219,34,629).  visible(lbpugt_4,629). happensAt(inactive(lbpugt_4),629).
orientation(lbpugt_2,132,630). coords(lbpugt_2,181,50,630).  visible(lbpugt_2,630). happensAt(active(lbpugt_2),630). orientation(lbpugt_4,135,630). coords(lbpugt_4,219,34,630).  visible(lbpugt_4,630). happensAt(inactive(lbpugt_4),630).
orientation(lbpugt_2,132,631). coords(lbpugt_2,181,50,631).  visible(lbpugt_2,631). happensAt(active(lbpugt_2),631). orientation(lbpugt_4,135,631). coords(lbpugt_4,219,34,631).  visible(lbpugt_4,631). happensAt(inactive(lbpugt_4),631).
orientation(lbpugt_2,132,632). coords(lbpugt_2,181,50,632).  visible(lbpugt_2,632). happensAt(active(lbpugt_2),632). orientation(lbpugt_4,135,632). coords(lbpugt_4,219,34,632).  visible(lbpugt_4,632). happensAt(inactive(lbpugt_4),632).
orientation(lbpugt_2,132,633). coords(lbpugt_2,182,51,633).  visible(lbpugt_2,633). happensAt(active(lbpugt_2),633). orientation(lbpugt_4,135,633). coords(lbpugt_4,219,34,633).  visible(lbpugt_4,633). happensAt(inactive(lbpugt_4),633).
orientation(lbpugt_2,132,634). coords(lbpugt_2,182,52,634).  visible(lbpugt_2,634). happensAt(active(lbpugt_2),634). orientation(lbpugt_4,135,634). coords(lbpugt_4,219,34,634).  visible(lbpugt_4,634). happensAt(inactive(lbpugt_4),634).
orientation(lbpugt_2,132,635). coords(lbpugt_2,182,52,635).  visible(lbpugt_2,635). happensAt(active(lbpugt_2),635). orientation(lbpugt_4,135,635). coords(lbpugt_4,219,34,635).  visible(lbpugt_4,635). happensAt(inactive(lbpugt_4),635).
orientation(lbpugt_2,132,636). coords(lbpugt_2,182,53,636).  visible(lbpugt_2,636). happensAt(active(lbpugt_2),636). orientation(lbpugt_4,135,636). coords(lbpugt_4,219,34,636).  visible(lbpugt_4,636). happensAt(inactive(lbpugt_4),636).
orientation(lbpugt_2,132,637). coords(lbpugt_2,182,53,637).  visible(lbpugt_2,637). happensAt(active(lbpugt_2),637). orientation(lbpugt_4,135,637). coords(lbpugt_4,219,34,637).  visible(lbpugt_4,637). happensAt(inactive(lbpugt_4),637).
orientation(lbpugt_2,132,638). coords(lbpugt_2,182,54,638).  visible(lbpugt_2,638). happensAt(active(lbpugt_2),638). orientation(lbpugt_4,135,638). coords(lbpugt_4,219,34,638).  visible(lbpugt_4,638). happensAt(inactive(lbpugt_4),638).
orientation(lbpugt_2,132,639). coords(lbpugt_2,183,54,639).  visible(lbpugt_2,639). happensAt(active(lbpugt_2),639). orientation(lbpugt_4,135,639). coords(lbpugt_4,219,34,639).  visible(lbpugt_4,639). happensAt(inactive(lbpugt_4),639).
orientation(lbpugt_2,132,640). coords(lbpugt_2,183,55,640).  visible(lbpugt_2,640). happensAt(active(lbpugt_2),640). orientation(lbpugt_4,135,640). coords(lbpugt_4,219,34,640).  visible(lbpugt_4,640). happensAt(inactive(lbpugt_4),640).
orientation(lbpugt_2,132,641). coords(lbpugt_2,183,55,641).  visible(lbpugt_2,641). happensAt(active(lbpugt_2),641). orientation(lbpugt_4,135,641). coords(lbpugt_4,219,34,641).  visible(lbpugt_4,641). happensAt(inactive(lbpugt_4),641).
orientation(lbpugt_2,132,642). coords(lbpugt_2,184,55,642).  visible(lbpugt_2,642). happensAt(active(lbpugt_2),642). orientation(lbpugt_4,135,642). coords(lbpugt_4,219,34,642).  visible(lbpugt_4,642). happensAt(inactive(lbpugt_4),642).
orientation(lbpugt_2,132,643). coords(lbpugt_2,184,56,643).  visible(lbpugt_2,643). happensAt(walking(lbpugt_2),643). orientation(lbpugt_4,135,643). coords(lbpugt_4,219,34,643).  visible(lbpugt_4,643). happensAt(inactive(lbpugt_4),643).
orientation(lbpugt_2,132,644). coords(lbpugt_2,184,56,644).  visible(lbpugt_2,644). happensAt(walking(lbpugt_2),644). orientation(lbpugt_4,135,644). coords(lbpugt_4,219,34,644).  visible(lbpugt_4,644). happensAt(inactive(lbpugt_4),644).
orientation(lbpugt_2,132,645). coords(lbpugt_2,184,57,645).  visible(lbpugt_2,645). happensAt(walking(lbpugt_2),645). orientation(lbpugt_4,135,645). coords(lbpugt_4,219,34,645).  visible(lbpugt_4,645). happensAt(inactive(lbpugt_4),645).
orientation(lbpugt_2,132,646). coords(lbpugt_2,184,57,646).  visible(lbpugt_2,646). happensAt(walking(lbpugt_2),646). orientation(lbpugt_4,135,646). coords(lbpugt_4,219,34,646).  visible(lbpugt_4,646). happensAt(inactive(lbpugt_4),646).
orientation(lbpugt_2,132,647). coords(lbpugt_2,184,57,647).  visible(lbpugt_2,647). happensAt(walking(lbpugt_2),647). orientation(lbpugt_4,135,647). coords(lbpugt_4,219,34,647).  visible(lbpugt_4,647). happensAt(inactive(lbpugt_4),647).
#show holdsAt/2.