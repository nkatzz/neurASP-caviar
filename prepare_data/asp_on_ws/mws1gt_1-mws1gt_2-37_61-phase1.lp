
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

	orientation(mws1gt_1,170,37). coords(mws1gt_1,65,193,37).  visible(mws1gt_1,37). happensAt(inactive(mws1gt_1),37). orientation(mws1gt_2,155,37). coords(mws1gt_2,44,136,37).  visible(mws1gt_2,37). happensAt(walking(mws1gt_2),37).
orientation(mws1gt_1,170,38). coords(mws1gt_1,65,193,38).  visible(mws1gt_1,38). happensAt(inactive(mws1gt_1),38). orientation(mws1gt_2,155,38). coords(mws1gt_2,43,137,38).  visible(mws1gt_2,38). happensAt(walking(mws1gt_2),38).
orientation(mws1gt_1,170,39). coords(mws1gt_1,65,193,39).  visible(mws1gt_1,39). happensAt(inactive(mws1gt_1),39). orientation(mws1gt_2,155,39). coords(mws1gt_2,43,138,39).  visible(mws1gt_2,39). happensAt(walking(mws1gt_2),39).
orientation(mws1gt_1,170,40). coords(mws1gt_1,65,193,40).  visible(mws1gt_1,40). happensAt(inactive(mws1gt_1),40). orientation(mws1gt_2,155,40). coords(mws1gt_2,43,139,40).  visible(mws1gt_2,40). happensAt(walking(mws1gt_2),40).
orientation(mws1gt_1,170,41). coords(mws1gt_1,65,193,41).  visible(mws1gt_1,41). happensAt(inactive(mws1gt_1),41). orientation(mws1gt_2,155,41). coords(mws1gt_2,43,140,41).  visible(mws1gt_2,41). happensAt(walking(mws1gt_2),41).
orientation(mws1gt_1,170,42). coords(mws1gt_1,65,193,42).  visible(mws1gt_1,42). happensAt(inactive(mws1gt_1),42). orientation(mws1gt_2,155,42). coords(mws1gt_2,43,140,42).  visible(mws1gt_2,42). happensAt(walking(mws1gt_2),42).
orientation(mws1gt_1,170,43). coords(mws1gt_1,65,193,43).  visible(mws1gt_1,43). happensAt(inactive(mws1gt_1),43). orientation(mws1gt_2,163,43). coords(mws1gt_2,42,140,43).  visible(mws1gt_2,43). happensAt(walking(mws1gt_2),43).
orientation(mws1gt_1,170,44). coords(mws1gt_1,65,193,44).  visible(mws1gt_1,44). happensAt(inactive(mws1gt_1),44). orientation(mws1gt_2,165,44). coords(mws1gt_2,41,140,44).  visible(mws1gt_2,44). happensAt(walking(mws1gt_2),44).
orientation(mws1gt_1,170,45). coords(mws1gt_1,65,193,45).  visible(mws1gt_1,45). happensAt(inactive(mws1gt_1),45). orientation(mws1gt_2,165,45). coords(mws1gt_2,41,142,45).  visible(mws1gt_2,45). happensAt(walking(mws1gt_2),45).
orientation(mws1gt_1,170,46). coords(mws1gt_1,65,193,46).  visible(mws1gt_1,46). happensAt(inactive(mws1gt_1),46). orientation(mws1gt_2,165,46). coords(mws1gt_2,41,143,46).  visible(mws1gt_2,46). happensAt(walking(mws1gt_2),46).
orientation(mws1gt_1,170,47). coords(mws1gt_1,65,193,47).  visible(mws1gt_1,47). happensAt(inactive(mws1gt_1),47). orientation(mws1gt_2,165,47). coords(mws1gt_2,41,146,47).  visible(mws1gt_2,47). happensAt(walking(mws1gt_2),47).
orientation(mws1gt_1,170,48). coords(mws1gt_1,65,193,48).  visible(mws1gt_1,48). happensAt(inactive(mws1gt_1),48). orientation(mws1gt_2,165,48). coords(mws1gt_2,41,147,48).  visible(mws1gt_2,48). happensAt(walking(mws1gt_2),48).
orientation(mws1gt_1,170,49). coords(mws1gt_1,65,193,49).  visible(mws1gt_1,49). happensAt(inactive(mws1gt_1),49). orientation(mws1gt_2,165,49). coords(mws1gt_2,41,147,49).  visible(mws1gt_2,49). happensAt(walking(mws1gt_2),49).
orientation(mws1gt_1,170,50). coords(mws1gt_1,65,193,50).  visible(mws1gt_1,50). happensAt(inactive(mws1gt_1),50). orientation(mws1gt_2,165,50). coords(mws1gt_2,41,147,50).  visible(mws1gt_2,50). happensAt(walking(mws1gt_2),50).
orientation(mws1gt_1,170,51). coords(mws1gt_1,65,193,51).  visible(mws1gt_1,51). happensAt(inactive(mws1gt_1),51). orientation(mws1gt_2,165,51). coords(mws1gt_2,41,147,51).  visible(mws1gt_2,51). happensAt(walking(mws1gt_2),51).
orientation(mws1gt_1,170,52). coords(mws1gt_1,65,193,52).  visible(mws1gt_1,52). happensAt(walking(mws1gt_1),52). orientation(mws1gt_2,165,52). coords(mws1gt_2,41,147,52).  visible(mws1gt_2,52). happensAt(walking(mws1gt_2),52).
orientation(mws1gt_1,170,53). coords(mws1gt_1,65,193,53).  visible(mws1gt_1,53). happensAt(walking(mws1gt_1),53). orientation(mws1gt_2,165,53). coords(mws1gt_2,41,147,53).  visible(mws1gt_2,53). happensAt(walking(mws1gt_2),53).
orientation(mws1gt_1,170,54). coords(mws1gt_1,65,193,54).  visible(mws1gt_1,54). happensAt(walking(mws1gt_1),54). orientation(mws1gt_2,165,54). coords(mws1gt_2,41,147,54).  visible(mws1gt_2,54). happensAt(walking(mws1gt_2),54).
orientation(mws1gt_1,170,55). coords(mws1gt_1,69,193,55).  visible(mws1gt_1,55). happensAt(walking(mws1gt_1),55). orientation(mws1gt_2,165,55). coords(mws1gt_2,41,147,55).  visible(mws1gt_2,55). happensAt(walking(mws1gt_2),55).
orientation(mws1gt_1,172,56). coords(mws1gt_1,69,193,56).  visible(mws1gt_1,56). happensAt(walking(mws1gt_1),56). orientation(mws1gt_2,165,56). coords(mws1gt_2,41,151,56).  visible(mws1gt_2,56). happensAt(walking(mws1gt_2),56).
orientation(mws1gt_1,172,57). coords(mws1gt_1,69,193,57).  visible(mws1gt_1,57). happensAt(walking(mws1gt_1),57). orientation(mws1gt_2,165,57). coords(mws1gt_2,41,154,57).  visible(mws1gt_2,57). happensAt(walking(mws1gt_2),57).
orientation(mws1gt_1,172,58). coords(mws1gt_1,70,193,58).  visible(mws1gt_1,58). happensAt(walking(mws1gt_1),58). orientation(mws1gt_2,165,58). coords(mws1gt_2,41,155,58).  visible(mws1gt_2,58). happensAt(walking(mws1gt_2),58).
orientation(mws1gt_1,172,59). coords(mws1gt_1,70,192,59).  visible(mws1gt_1,59). happensAt(walking(mws1gt_1),59). orientation(mws1gt_2,165,59). coords(mws1gt_2,39,155,59).  visible(mws1gt_2,59). happensAt(walking(mws1gt_2),59).
orientation(mws1gt_1,172,60). coords(mws1gt_1,71,192,60).  visible(mws1gt_1,60). happensAt(walking(mws1gt_1),60). orientation(mws1gt_2,165,60). coords(mws1gt_2,39,157,60).  visible(mws1gt_2,60). happensAt(walking(mws1gt_2),60).
orientation(mws1gt_1,172,61). coords(mws1gt_1,71,192,61).  visible(mws1gt_1,61). happensAt(walking(mws1gt_1),61). orientation(mws1gt_2,165,61). coords(mws1gt_2,39,157,61).  visible(mws1gt_2,61). happensAt(walking(mws1gt_2),61).
#show holdsAt/2.