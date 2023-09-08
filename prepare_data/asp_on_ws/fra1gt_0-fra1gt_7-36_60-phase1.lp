
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

	orientation(fra1gt_0,156,36). coords(fra1gt_0,217,174,36).  visible(fra1gt_0,36). happensAt(walking(fra1gt_0),36). orientation(fra1gt_7,136,36). coords(fra1gt_7,145,40,36).  visible(fra1gt_7,36). happensAt(walking(fra1gt_7),36).
orientation(fra1gt_0,156,37). coords(fra1gt_0,219,176,37).  visible(fra1gt_0,37). happensAt(walking(fra1gt_0),37). orientation(fra1gt_7,136,37). coords(fra1gt_7,145,40,37).  visible(fra1gt_7,37). happensAt(walking(fra1gt_7),37).
orientation(fra1gt_0,162,38). coords(fra1gt_0,221,178,38).  visible(fra1gt_0,38). happensAt(walking(fra1gt_0),38). orientation(fra1gt_7,136,38). coords(fra1gt_7,145,40,38).  visible(fra1gt_7,38). happensAt(walking(fra1gt_7),38).
orientation(fra1gt_0,162,39). coords(fra1gt_0,222,179,39).  visible(fra1gt_0,39). happensAt(walking(fra1gt_0),39). orientation(fra1gt_7,136,39). coords(fra1gt_7,145,40,39).  visible(fra1gt_7,39). happensAt(inactive(fra1gt_7),39).
orientation(fra1gt_0,162,40). coords(fra1gt_0,224,179,40).  visible(fra1gt_0,40). happensAt(walking(fra1gt_0),40). orientation(fra1gt_7,136,40). coords(fra1gt_7,145,40,40).  visible(fra1gt_7,40). happensAt(inactive(fra1gt_7),40).
orientation(fra1gt_0,162,41). coords(fra1gt_0,225,182,41).  visible(fra1gt_0,41). happensAt(walking(fra1gt_0),41). orientation(fra1gt_7,136,41). coords(fra1gt_7,145,40,41).  visible(fra1gt_7,41). happensAt(inactive(fra1gt_7),41).
orientation(fra1gt_0,162,42). coords(fra1gt_0,227,186,42).  visible(fra1gt_0,42). happensAt(walking(fra1gt_0),42). orientation(fra1gt_7,136,42). coords(fra1gt_7,145,40,42).  visible(fra1gt_7,42). happensAt(inactive(fra1gt_7),42).
orientation(fra1gt_0,162,43). coords(fra1gt_0,228,190,43).  visible(fra1gt_0,43). happensAt(walking(fra1gt_0),43). orientation(fra1gt_7,136,43). coords(fra1gt_7,145,40,43).  visible(fra1gt_7,43). happensAt(inactive(fra1gt_7),43).
orientation(fra1gt_0,162,44). coords(fra1gt_0,229,193,44).  visible(fra1gt_0,44). happensAt(walking(fra1gt_0),44). orientation(fra1gt_7,136,44). coords(fra1gt_7,145,40,44).  visible(fra1gt_7,44). happensAt(inactive(fra1gt_7),44).
orientation(fra1gt_0,167,45). coords(fra1gt_0,230,195,45).  visible(fra1gt_0,45). happensAt(walking(fra1gt_0),45). orientation(fra1gt_7,136,45). coords(fra1gt_7,145,40,45).  visible(fra1gt_7,45). happensAt(inactive(fra1gt_7),45).
orientation(fra1gt_0,167,46). coords(fra1gt_0,230,196,46).  visible(fra1gt_0,46). happensAt(walking(fra1gt_0),46). orientation(fra1gt_7,136,46). coords(fra1gt_7,144,40,46).  visible(fra1gt_7,46). happensAt(inactive(fra1gt_7),46).
orientation(fra1gt_0,176,47). coords(fra1gt_0,231,197,47).  visible(fra1gt_0,47). happensAt(walking(fra1gt_0),47). orientation(fra1gt_7,136,47). coords(fra1gt_7,144,40,47).  visible(fra1gt_7,47). happensAt(inactive(fra1gt_7),47).
orientation(fra1gt_0,6,48). coords(fra1gt_0,232,198,48).  visible(fra1gt_0,48). happensAt(walking(fra1gt_0),48). orientation(fra1gt_7,136,48). coords(fra1gt_7,144,41,48).  visible(fra1gt_7,48). happensAt(inactive(fra1gt_7),48).
orientation(fra1gt_0,6,49). coords(fra1gt_0,232,201,49).  visible(fra1gt_0,49). happensAt(walking(fra1gt_0),49). orientation(fra1gt_7,136,49). coords(fra1gt_7,144,41,49).  visible(fra1gt_7,49). happensAt(inactive(fra1gt_7),49).
orientation(fra1gt_0,6,50). coords(fra1gt_0,233,205,50).  visible(fra1gt_0,50). happensAt(walking(fra1gt_0),50). orientation(fra1gt_7,136,50). coords(fra1gt_7,144,41,50).  visible(fra1gt_7,50). happensAt(inactive(fra1gt_7),50).
orientation(fra1gt_0,4,51). coords(fra1gt_0,233,208,51).  visible(fra1gt_0,51). happensAt(walking(fra1gt_0),51). orientation(fra1gt_7,136,51). coords(fra1gt_7,144,41,51).  visible(fra1gt_7,51). happensAt(inactive(fra1gt_7),51).
orientation(fra1gt_0,9,52). coords(fra1gt_0,234,211,52).  visible(fra1gt_0,52). happensAt(walking(fra1gt_0),52). orientation(fra1gt_7,136,52). coords(fra1gt_7,144,41,52).  visible(fra1gt_7,52). happensAt(inactive(fra1gt_7),52).
orientation(fra1gt_0,9,53). coords(fra1gt_0,235,213,53).  visible(fra1gt_0,53). happensAt(walking(fra1gt_0),53). orientation(fra1gt_7,136,53). coords(fra1gt_7,144,41,53).  visible(fra1gt_7,53). happensAt(inactive(fra1gt_7),53).
orientation(fra1gt_0,9,54). coords(fra1gt_0,236,216,54).  visible(fra1gt_0,54). happensAt(walking(fra1gt_0),54). orientation(fra1gt_7,136,54). coords(fra1gt_7,144,41,54).  visible(fra1gt_7,54). happensAt(inactive(fra1gt_7),54).
orientation(fra1gt_0,5,55). coords(fra1gt_0,238,218,55).  visible(fra1gt_0,55). happensAt(walking(fra1gt_0),55). orientation(fra1gt_7,136,55). coords(fra1gt_7,144,41,55).  visible(fra1gt_7,55). happensAt(inactive(fra1gt_7),55).
orientation(fra1gt_0,5,56). coords(fra1gt_0,240,221,56).  visible(fra1gt_0,56). happensAt(walking(fra1gt_0),56). orientation(fra1gt_7,136,56). coords(fra1gt_7,144,41,56).  visible(fra1gt_7,56). happensAt(inactive(fra1gt_7),56).
orientation(fra1gt_0,5,57). coords(fra1gt_0,243,223,57).  visible(fra1gt_0,57). happensAt(walking(fra1gt_0),57). orientation(fra1gt_7,136,57). coords(fra1gt_7,144,41,57).  visible(fra1gt_7,57). happensAt(inactive(fra1gt_7),57).
orientation(fra1gt_0,5,58). coords(fra1gt_0,244,225,58).  visible(fra1gt_0,58). happensAt(walking(fra1gt_0),58). orientation(fra1gt_7,136,58). coords(fra1gt_7,144,41,58).  visible(fra1gt_7,58). happensAt(inactive(fra1gt_7),58).
orientation(fra1gt_0,5,59). coords(fra1gt_0,247,226,59).  visible(fra1gt_0,59). happensAt(walking(fra1gt_0),59). orientation(fra1gt_7,136,59). coords(fra1gt_7,144,41,59).  visible(fra1gt_7,59). happensAt(inactive(fra1gt_7),59).
orientation(fra1gt_0,5,60). coords(fra1gt_0,247,227,60).  visible(fra1gt_0,60). happensAt(walking(fra1gt_0),60). orientation(fra1gt_7,136,60). coords(fra1gt_7,144,41,60).  visible(fra1gt_7,60). happensAt(inactive(fra1gt_7),60).
#show holdsAt/2.