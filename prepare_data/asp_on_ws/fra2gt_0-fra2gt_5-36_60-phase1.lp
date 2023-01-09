
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

	orientation(fra2gt_0,154,36). coords(fra2gt_0,163,128,36).  visible(fra2gt_0,36). happensAt(walking(fra2gt_0),36). orientation(fra2gt_5,150,36). coords(fra2gt_5,44,105,36).  visible(fra2gt_5,36). happensAt(walking(fra2gt_5),36).
orientation(fra2gt_0,154,37). coords(fra2gt_0,162,129,37).  visible(fra2gt_0,37). happensAt(active(fra2gt_0),37). orientation(fra2gt_5,150,37). coords(fra2gt_5,44,105,37).  visible(fra2gt_5,37). happensAt(walking(fra2gt_5),37).
orientation(fra2gt_0,154,38). coords(fra2gt_0,162,129,38).  visible(fra2gt_0,38). happensAt(active(fra2gt_0),38). orientation(fra2gt_5,150,38). coords(fra2gt_5,44,105,38).  visible(fra2gt_5,38). happensAt(walking(fra2gt_5),38).
orientation(fra2gt_0,154,39). coords(fra2gt_0,161,129,39).  visible(fra2gt_0,39). happensAt(active(fra2gt_0),39). orientation(fra2gt_5,150,39). coords(fra2gt_5,44,105,39).  visible(fra2gt_5,39). happensAt(walking(fra2gt_5),39).
orientation(fra2gt_0,154,40). coords(fra2gt_0,160,129,40).  visible(fra2gt_0,40). happensAt(active(fra2gt_0),40). orientation(fra2gt_5,150,40). coords(fra2gt_5,43,105,40).  visible(fra2gt_5,40). happensAt(walking(fra2gt_5),40).
orientation(fra2gt_0,154,41). coords(fra2gt_0,159,129,41).  visible(fra2gt_0,41). happensAt(active(fra2gt_0),41). orientation(fra2gt_5,150,41). coords(fra2gt_5,43,105,41).  visible(fra2gt_5,41). happensAt(walking(fra2gt_5),41).
orientation(fra2gt_0,154,42). coords(fra2gt_0,159,129,42).  visible(fra2gt_0,42). happensAt(active(fra2gt_0),42). orientation(fra2gt_5,150,42). coords(fra2gt_5,43,106,42).  visible(fra2gt_5,42). happensAt(walking(fra2gt_5),42).
orientation(fra2gt_0,154,43). coords(fra2gt_0,158,129,43).  visible(fra2gt_0,43). happensAt(active(fra2gt_0),43). orientation(fra2gt_5,150,43). coords(fra2gt_5,43,106,43).  visible(fra2gt_5,43). happensAt(walking(fra2gt_5),43).
orientation(fra2gt_0,154,44). coords(fra2gt_0,158,129,44).  visible(fra2gt_0,44). happensAt(active(fra2gt_0),44). orientation(fra2gt_5,150,44). coords(fra2gt_5,43,106,44).  visible(fra2gt_5,44). happensAt(walking(fra2gt_5),44).
orientation(fra2gt_0,150,45). coords(fra2gt_0,158,129,45).  visible(fra2gt_0,45). happensAt(active(fra2gt_0),45). orientation(fra2gt_5,150,45). coords(fra2gt_5,43,107,45).  visible(fra2gt_5,45). happensAt(walking(fra2gt_5),45).
orientation(fra2gt_0,150,46). coords(fra2gt_0,158,129,46).  visible(fra2gt_0,46). happensAt(active(fra2gt_0),46). orientation(fra2gt_5,150,46). coords(fra2gt_5,41,107,46).  visible(fra2gt_5,46). happensAt(walking(fra2gt_5),46).
orientation(fra2gt_0,150,47). coords(fra2gt_0,158,129,47).  visible(fra2gt_0,47). happensAt(active(fra2gt_0),47). orientation(fra2gt_5,150,47). coords(fra2gt_5,39,106,47).  visible(fra2gt_5,47). happensAt(walking(fra2gt_5),47).
orientation(fra2gt_0,150,48). coords(fra2gt_0,158,129,48).  visible(fra2gt_0,48). happensAt(active(fra2gt_0),48). orientation(fra2gt_5,150,48). coords(fra2gt_5,39,106,48).  visible(fra2gt_5,48). happensAt(walking(fra2gt_5),48).
orientation(fra2gt_0,150,49). coords(fra2gt_0,158,129,49).  visible(fra2gt_0,49). happensAt(active(fra2gt_0),49). orientation(fra2gt_5,150,49). coords(fra2gt_5,39,106,49).  visible(fra2gt_5,49). happensAt(walking(fra2gt_5),49).
orientation(fra2gt_0,150,50). coords(fra2gt_0,158,129,50).  visible(fra2gt_0,50). happensAt(active(fra2gt_0),50). orientation(fra2gt_5,150,50). coords(fra2gt_5,39,106,50).  visible(fra2gt_5,50). happensAt(walking(fra2gt_5),50).
orientation(fra2gt_0,150,51). coords(fra2gt_0,158,129,51).  visible(fra2gt_0,51). happensAt(active(fra2gt_0),51). orientation(fra2gt_5,150,51). coords(fra2gt_5,38,107,51).  visible(fra2gt_5,51). happensAt(walking(fra2gt_5),51).
orientation(fra2gt_0,150,52). coords(fra2gt_0,157,129,52).  visible(fra2gt_0,52). happensAt(active(fra2gt_0),52). orientation(fra2gt_5,150,52). coords(fra2gt_5,38,107,52).  visible(fra2gt_5,52). happensAt(walking(fra2gt_5),52).
orientation(fra2gt_0,150,53). coords(fra2gt_0,157,129,53).  visible(fra2gt_0,53). happensAt(active(fra2gt_0),53). orientation(fra2gt_5,150,53). coords(fra2gt_5,38,107,53).  visible(fra2gt_5,53). happensAt(walking(fra2gt_5),53).
orientation(fra2gt_0,150,54). coords(fra2gt_0,157,129,54).  visible(fra2gt_0,54). happensAt(inactive(fra2gt_0),54). orientation(fra2gt_5,150,54). coords(fra2gt_5,38,107,54).  visible(fra2gt_5,54). happensAt(walking(fra2gt_5),54).
orientation(fra2gt_0,150,55). coords(fra2gt_0,157,129,55).  visible(fra2gt_0,55). happensAt(inactive(fra2gt_0),55). orientation(fra2gt_5,150,55). coords(fra2gt_5,37,108,55).  visible(fra2gt_5,55). happensAt(walking(fra2gt_5),55).
orientation(fra2gt_0,150,56). coords(fra2gt_0,157,129,56).  visible(fra2gt_0,56). happensAt(inactive(fra2gt_0),56). orientation(fra2gt_5,150,56). coords(fra2gt_5,37,109,56).  visible(fra2gt_5,56). happensAt(walking(fra2gt_5),56).
orientation(fra2gt_0,150,57). coords(fra2gt_0,157,129,57).  visible(fra2gt_0,57). happensAt(inactive(fra2gt_0),57). orientation(fra2gt_5,150,57). coords(fra2gt_5,37,109,57).  visible(fra2gt_5,57). happensAt(walking(fra2gt_5),57).
orientation(fra2gt_0,150,58). coords(fra2gt_0,157,129,58).  visible(fra2gt_0,58). happensAt(inactive(fra2gt_0),58). orientation(fra2gt_5,150,58). coords(fra2gt_5,36,110,58).  visible(fra2gt_5,58). happensAt(walking(fra2gt_5),58).
orientation(fra2gt_0,150,59). coords(fra2gt_0,157,129,59).  visible(fra2gt_0,59). happensAt(inactive(fra2gt_0),59). orientation(fra2gt_5,150,59). coords(fra2gt_5,35,110,59).  visible(fra2gt_5,59). happensAt(walking(fra2gt_5),59).
orientation(fra2gt_0,150,60). coords(fra2gt_0,157,129,60).  visible(fra2gt_0,60). happensAt(inactive(fra2gt_0),60). orientation(fra2gt_5,150,60). coords(fra2gt_5,35,110,60).  visible(fra2gt_5,60). happensAt(walking(fra2gt_5),60).
#show holdsAt/2.